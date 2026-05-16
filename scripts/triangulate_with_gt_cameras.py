#!/usr/bin/env python3
"""
Triangulation using ground-truth camera files (Strecha / PMVS-style .camera)

This script:
 - reads images from an images directory
 - loads per-image camera files from a gt cameras directory (default: images_dir/gt_dense_cameras)
   which follow the Strecha / PMVS-style layout (K rows, distortion, R rows, camera center, image size)
 - detects features (SIFT preferred, ORB fallback), matches descriptors, builds multi-image tracks
 - triangulates tracks using the GT projection matrices P = K [R | t] (t = -R @ C)
 - optionally performs per-point non-linear LM refinement (camera poses fixed)
 - saves a multi-image matches visualization and multiple 3D perspective views (rotating azimuth)

Example:
  python3 scripts/triangulate_with_gt_cameras.py \
      --images_dir Benchmarking_Camera_Calibration_2008/fountain-P11/images \
      --images 0000.jpg 0001.jpg 0002.jpg --out outputs/triangulation_gt.png
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_strecha_camera(path):
    """Read Strecha/PMVS-style .camera file.

    Expected (typical) layout (one row per line):
      lines 0-2: K (3 rows)
      line 3: distortion (often 3 zeros)
      lines 4-6: R (3 rows)
      line 7: camera center C (3 values)
      line 8: image size (w h)

    Returns: K (3x3), distortion (list or None), R (3x3), t (3x1 column), C (3x1), img_wh (tuple or None)
    """
    txt = open(path, 'r').read().strip().splitlines()
    lines = [l.strip() for l in txt if l.strip()]
    if not lines:
        raise ValueError('Empty camera file: %s' % path)

    # try the common multi-line format first
    if len(lines) >= 8:
        def to_floats(s):
            return list(map(float, s.replace(',', ' ').split()))

        K_rows = [to_floats(lines[i]) for i in range(3)]
        K = np.array(K_rows, dtype=float)
        # distortion may be present on a single line
        distortion = to_floats(lines[3]) if len(to_floats(lines[3])) >= 1 else None
        R_rows = [to_floats(lines[i]) for i in range(4, 7)]
        R = np.array(R_rows, dtype=float)
        C_vals = to_floats(lines[7])
        C = np.array(C_vals, dtype=float).reshape(3, 1)
        img_wh = None
        if len(lines) >= 9:
            try:
                wh = list(map(int, lines[8].split()))
                if len(wh) >= 2:
                    img_wh = (wh[0], wh[1])
            except Exception:
                img_wh = None
        # compute translation t = -R * C (world -> camera)
        t = -R.dot(C)
        return K, distortion, R, t.reshape(3, 1), C, img_wh

    # fallback: parse numeric floats and interpret as a projection matrix
    nums = list(map(float, ' '.join(lines).replace(',', ' ').split()))
    if len(nums) == 12:
        P = np.array(nums, dtype=float).reshape(3, 4)
        Kc, R, trans = cv2.decomposeProjectionMatrix(P)
        # normalize K so that K[2,2] == 1
        Kc = Kc / Kc[2, 2]
        C = (trans[:3] / trans[3]).reshape(3, 1)
        t = -R.dot(C)
        return Kc, None, R, t.reshape(3, 1), C, None
    if len(nums) == 16:
        P4 = np.array(nums, dtype=float).reshape(4, 4)
        P = P4[:3, :4]
        Kc, R, trans = cv2.decomposeProjectionMatrix(P)
        Kc = Kc / Kc[2, 2]
        C = (trans[:3] / trans[3]).reshape(3, 1)
        t = -R.dot(C)
        return Kc, None, R, t.reshape(3, 1), C, None

    raise ValueError('Unrecognized camera file format: %s (read %d numbers)' % (path, len(nums)))


def detect_and_compute(img, use_sift=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if use_sift:
        try:
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(gray, None)
            if des is not None:
                return kp, des, 'SIFT'
        except Exception:
            pass
    orb = cv2.ORB_create(4000)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des, 'ORB'


def match_descriptors(des1, des2, method):
    if des1 is None or des2 is None:
        return []
    if method == 'SIFT':
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
        if des2.dtype != np.float32:
            des2 = des2.astype(np.float32)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        knn = flann.knnMatch(des1, des2, k=2)
        ratio_thresh = 0.7
        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        return good
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)
        ratio_thresh = 0.85
        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio_thresh * n.distance:
                good.append(m)
        return good


class UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.add(x)
        self.add(y)
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            return
        self.parent[ry] = rx


def triangulate_with_gt(images_dir, image_names, gt_dir, out_path,
                        method='linear', refine_iters=20, n_views=12, view_elev=20.0,
                        vis_keep_percentile=95.0):
    if len(image_names) < 2:
        raise ValueError('Need at least two images')

    # Load GT cameras
    poses_R = [None] * len(image_names)
    poses_t = [None] * len(image_names)
    Ks = [None] * len(image_names)
    Cs = [None] * len(image_names)

    for i, name in enumerate(image_names):
        # try several candidate file names in gt_dir
        candidates = [
            os.path.join(gt_dir, name + '.camera'),
            os.path.join(gt_dir, name),
            os.path.join(gt_dir, os.path.splitext(name)[0] + '.camera'),
            os.path.join(gt_dir, os.path.splitext(name)[0] + '.txt'),
        ]
        cam_file = None
        for c in candidates:
            if os.path.exists(c):
                cam_file = c
                break
        if cam_file is None:
            # no GT camera for this image
            continue
        try:
            K, distortion, R, t, C, img_wh = read_strecha_camera(cam_file)
            Ks[i] = K
            poses_R[i] = R
            poses_t[i] = t.reshape(3, 1)
            Cs[i] = C
            print(f'Loaded GT camera for image {name} from {cam_file}')
        except Exception as e:
            print('Warning: failed to parse GT camera for %s: %s' % (name, str(e)))

    # Read images and compute features
    imgs = []
    kps = []
    dess = []
    methods = []
    for name in image_names:
        path = os.path.join(images_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError('Image not found: %s' % path)
        img = cv2.imread(path)
        imgs.append(img)
        kp, des, detector = detect_and_compute(img)
        kps.append(kp)
        dess.append(des)
        methods.append(detector)

    # For images without a K loaded, try to fall back to images_dir/K.txt
    default_K = None
    kfile = os.path.join(images_dir, 'K.txt')
    if os.path.exists(kfile):
        try:
            vals = []
            for line in open(kfile, 'r'):
                for p in line.replace(';', ' ').split():
                    try:
                        vals.append(float(p))
                    except Exception:
                        pass
            if len(vals) >= 9:
                default_K = np.array(vals[:9], dtype=float).reshape(3, 3)
        except Exception:
            default_K = None
    # fill missing Ks with default_K
    for i in range(len(Ks)):
        if Ks[i] is None and default_K is not None:
            Ks[i] = default_K

    # Pairwise matching & union-find tracks
    uf = UnionFind()
    matches_by_pair = {}
    print('Matching descriptors between image pairs...')
    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            if methods[i] != methods[j]:
                matches_by_pair[(i, j)] = []
                continue
            good = match_descriptors(dess[i], dess[j], methods[i])
            matches_by_pair[(i, j)] = good
            for m in good:
                a = (i, m.queryIdx)
                b = (j, m.trainIdx)
                uf.add(a)
                uf.add(b)
                uf.union(a, b)

    # Build tracks
    tracks = {}
    for key in uf.parent.keys():
        root = uf.find(key)
        tracks.setdefault(root, set()).add(key)

    good_tracks = []
    for tr in tracks.values():
        obs = list(tr)
        imgs_seen = set([i for (i, k) in obs])
        if len(imgs_seen) >= 2:
            good_tracks.append(obs)

    print('Built %d tracks (%d with >=2 views)' % (len(tracks), len(good_tracks)))

    # Triangulate using GT poses (for observations that have known poses/Ks)
    points3d = []
    tracks_used = []
    # debugging counters for why tracks might be rejected
    cnt_total = len(good_tracks)
    cnt_has_pose = 0
    cnt_svd_ok = 0
    cnt_valid_homog = 0
    cnt_positive_depth = 0
    cnt_bad_homog = 0
    cnt_svd_fail = 0

    for tr in good_tracks:
        obs = list(tr)
        Pi_list = []
        uvs = []
        img_inds = []
        for (i, kp_idx) in obs:
            if poses_R[i] is None or poses_t[i] is None or Ks[i] is None:
                continue
            Ri = poses_R[i]
            ti = poses_t[i]
            Ki = Ks[i]
            Pi = Ki.dot(np.hstack((Ri, ti)))
            Pi_list.append(Pi)
            u, v = kps[i][kp_idx].pt
            uvs.append((u, v))
            img_inds.append(i)
        if len(Pi_list) < 2:
            # not enough views with GT pose/K for this track
            continue
        cnt_has_pose += 1

        # build DLT system
        M = []
        for Pi, (x, y) in zip(Pi_list, uvs):
            P0 = Pi[0, :]
            P1 = Pi[1, :]
            P2 = Pi[2, :]
            M.append(x * P2 - P0)
            M.append(y * P2 - P1)
        M = np.array(M)
        try:
            _, _, Vt = np.linalg.svd(M)
            cnt_svd_ok += 1
        except np.linalg.LinAlgError:
            cnt_svd_fail += 1
            continue
        Xh = Vt[-1, :]
        if abs(Xh[-1]) < 1e-8:
            cnt_bad_homog += 1
            continue
        X = Xh[:3] / Xh[3]
        cnt_valid_homog += 1

        # check positive depth in at least two cameras using the GT poses (not K*R)
        pos_depths = 0
        for ii in img_inds:
            Ri = poses_R[ii]
            ti = poses_t[ii]
            z = float(Ri[2, :].dot(X) + ti[2, 0])
            if z > 1e-6:
                pos_depths += 1
        if pos_depths < 2:
            # reconstructed point lies behind most cameras
            continue
        cnt_positive_depth += 1

        points3d.append(X)
        tracks_used.append({'obs': obs, 'Pi_list': Pi_list, 'uvs': uvs})

    if len(points3d) == 0:
        # print some debug stats to help diagnose why no points were kept
        try:
            print('Debug counts: total_tracks=%d, has_pose=%d, svd_ok=%d, svd_fail=%d, bad_homog=%d, valid_homog=%d, positive_depth=%d' % (
                cnt_total, cnt_has_pose, cnt_svd_ok, cnt_svd_fail, cnt_bad_homog, cnt_valid_homog, cnt_positive_depth))
        except Exception:
            pass
        raise RuntimeError('No 3D points reconstructed')

    Xf = np.array(points3d)
    print('Triangulated %d points from %d tracks' % (Xf.shape[0], len(tracks_used)))

    # reprojection and optional nonlinear refinement
    def reprojection_errors_point(X, Pi_list, uvs):
        errs = []
        for Pi, (u_obs, v_obs) in zip(Pi_list, uvs):
            P0 = Pi[0, :3]; p04 = Pi[0, 3]
            P1 = Pi[1, :3]; p14 = Pi[1, 3]
            P2 = Pi[2, :3]; p24 = Pi[2, 3]
            Nu = float(P0.dot(X) + p04)
            Nv = float(P1.dot(X) + p14)
            D = float(P2.dot(X) + p24)
            if abs(D) < 1e-12:
                errs.append(1e6)
                continue
            u_proj = Nu / D
            v_proj = Nv / D
            errs.append(np.hypot(u_proj - u_obs, v_proj - v_obs))
        return np.array(errs, dtype=float)

    def refine_point_lm(X0, Pi_list, uvs, max_iter=20, tol=1e-6):
        X = np.array(X0, dtype=float).reshape(3)
        errs = reprojection_errors_point(X, Pi_list, uvs)
        cost = float((errs ** 2).sum())
        lm_lambda = 1e-3
        for it in range(max_iter):
            J_rows = []
            r = []
            bad = False
            for Pi, (u_obs, v_obs) in zip(Pi_list, uvs):
                n0 = Pi[0, :3]; n0_4 = Pi[0, 3]
                n1 = Pi[1, :3]; n1_4 = Pi[1, 3]
                n2 = Pi[2, :3]; n2_4 = Pi[2, 3]
                Nu = float(n0.dot(X) + n0_4)
                Nv = float(n1.dot(X) + n1_4)
                D = float(n2.dot(X) + n2_4)
                if abs(D) < 1e-12:
                    bad = True
                    break
                u_proj = Nu / D
                v_proj = Nv / D
                ru = u_proj - u_obs
                rv = v_proj - v_obs
                du_dX = (n0 * D - Nu * n2) / (D * D)
                dv_dX = (n1 * D - Nv * n2) / (D * D)
                J_rows.append(du_dX)
                J_rows.append(dv_dX)
                r.append(ru)
                r.append(rv)
            if bad:
                break
            J = np.vstack(J_rows)
            r = np.array(r, dtype=float)
            H = J.T.dot(J)
            g = J.T.dot(r)
            H_lm = H + lm_lambda * np.diag(np.diag(H))
            try:
                dx = -np.linalg.solve(H_lm, g)
            except np.linalg.LinAlgError:
                try:
                    dx, _, _, _ = np.linalg.lstsq(H_lm, -g, rcond=None)
                except Exception:
                    break
            if np.linalg.norm(dx) < tol:
                X = X + dx
                break
            X_new = X + dx
            errs_new = reprojection_errors_point(X_new, Pi_list, uvs)
            cost_new = float((errs_new ** 2).sum())
            if cost_new < cost:
                X = X_new
                cost = cost_new
                lm_lambda = max(lm_lambda * 0.1, 1e-12)
            else:
                lm_lambda *= 10.0
        final_errs = reprojection_errors_point(X, Pi_list, uvs)
        return X, final_errs

    reproj_before = []
    reproj_after = []
    refined_points = []
    for X, info in zip(points3d, tracks_used):
        Pi_list = info.get('Pi_list', [])
        uvs = info.get('uvs', [])
        if len(Pi_list) < 2:
            continue
        errs0 = reprojection_errors_point(X, Pi_list, uvs)
        reproj_before.append(errs0.mean())
        if method == 'nonlinear':
            Xr, errs_r = refine_point_lm(X, Pi_list, uvs, max_iter=refine_iters)
            refined_points.append(Xr)
            reproj_after.append(errs_r.mean())
        else:
            refined_points.append(X)
            reproj_after.append(errs0.mean())

    reproj_before = np.array(reproj_before) if len(reproj_before) else np.array([])
    reproj_after = np.array(reproj_after) if len(reproj_after) else np.array([])
    if reproj_before.size:
        print('Reprojection error (pixels) - linear (mean): %.4f, median: %.4f, rms: %.4f' % (
            reproj_before.mean(), np.median(reproj_before), np.sqrt((reproj_before ** 2).mean())))
    if reproj_after.size:
        print('Reprojection error (pixels) - after %s refinement (mean): %.4f, median: %.4f, rms: %.4f' % (
            ('non-linear' if method == 'nonlinear' else 'linear'), reproj_after.mean(), np.median(reproj_after), np.sqrt((reproj_after ** 2).mean())))

    Xf = np.array(refined_points)

    # Prepare a filtered point set for visualization so that a few extreme outliers
    # do not determine the plot bounds. We prefer filtering by reprojection error
    # (if available), otherwise fall back to per-axis percentile cropping.
    Xf_vis = Xf
    try:
        if Xf.size != 0 and reproj_after.size == Xf.shape[0]:
            p = float(vis_keep_percentile)
            if 0 < p < 100:
                thr = np.percentile(reproj_after, p)
                mask = (reproj_after <= thr)
                Xf_vis = Xf[mask]
                if Xf_vis.size == 0:
                    Xf_vis = Xf
                print(f'Visualization filter: kept {Xf_vis.shape[0]} of {Xf.shape[0]} points (<= {p}th percentile reproj error <= {thr:.2f}px)')
        elif Xf.size != 0:
            # Fallback: keep central vis_keep_percentile percent of points per-axis
            lowp = max(0.0, (100.0 - float(vis_keep_percentile)) / 2.0)
            highp = min(100.0, 100.0 - lowp)
            mins = np.percentile(Xf, lowp, axis=0)
            maxs = np.percentile(Xf, highp, axis=0)
            mask = np.all((Xf >= mins) & (Xf <= maxs), axis=1)
            Xf_vis = Xf[mask]
            if Xf_vis.size == 0:
                Xf_vis = Xf
            print(f'Visualization filter (axis percentiles {lowp}-{highp}): kept {Xf_vis.shape[0]} of {Xf.shape[0]} points')
    except Exception as e:
        print('Warning: visualization filtering failed: %s' % str(e))
        Xf_vis = Xf
    # save multi-image matches visualization
    try:
        heights = [img.shape[0] for img in imgs]
        max_h = max(heights)
        total_w = sum([img.shape[1] for img in imgs])
        vis = np.zeros((max_h, total_w, 3), dtype=np.uint8)
        x_offsets = []
        ox = 0
        for idx, img in enumerate(imgs):
            img_vis = img.copy()
            if img_vis.ndim == 2:
                img_vis = cv2.cvtColor(img_vis, cv2.COLOR_GRAY2BGR)
            h, w = img_vis.shape[:2]
            vis[:h, ox:ox + w] = img_vis
            x_offsets.append(ox)
            ox += w
        for (i, j), matches in matches_by_pair.items():
            for m in matches:
                x1, y1 = kps[i][m.queryIdx].pt
                x2, y2 = kps[j][m.trainIdx].pt
                p1 = (int(round(x_offsets[i] + x1)), int(round(y1)))
                p2 = (int(round(x_offsets[j] + x2)), int(round(y2)))
                cv2.line(vis, p1, p2, (150, 150, 150), 1, lineType=cv2.LINE_AA)
        for info in tracks_used:
            obs = info.get('obs', [])
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))
            for (i, kp_idx) in obs:
                x, y = kps[i][kp_idx].pt
                p = (int(round(x_offsets[i] + x)), int(round(y)))
                cv2.circle(vis, p, 3, color, -1, lineType=cv2.LINE_AA)
        matches_out = os.path.splitext(out_path)[0] + '_matches_gt.png'
        cv2.imwrite(matches_out, vis)
        print('Saved multi-image matches visualization to %s' % matches_out)
    except Exception as e:
        print('Warning: failed to save multi-image matches visualization: %s' % str(e))

    # 3D visualization and multiple perspective saves
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if Xf_vis.size == 0:
        print('No points to visualize')
    else:
        ax.scatter(Xf_vis[:, 0], Xf_vis[:, 1], Xf_vis[:, 2], s=1, c='b', alpha=0.8)

    # camera centers
    for i in range(len(image_names)):
        if poses_R[i] is None or poses_t[i] is None:
            continue
        R = poses_R[i]
        t = poses_t[i]
        C = -R.T.dot(t).ravel()
        ax.scatter(C[0], C[1], C[2], s=50, marker='^')
        ax.text(C[0], C[1], C[2], 'Cam%d' % i)

    ax.scatter(0, 0, 0, c='r', s=60, marker='x')

    if Xf_vis.size == 0:
        X_vals = np.zeros((1, 3))
    else:
        X_vals = Xf_vis
    max_range = np.array([X_vals[:, 0].max() - X_vals[:, 0].min(),
                          X_vals[:, 1].max() - X_vals[:, 1].min(),
                          X_vals[:, 2].max() - X_vals[:, 2].min()]).max() / 2.0
    mid_x = (X_vals[:, 0].max() + X_vals[:, 0].min()) * 0.5
    mid_y = (X_vals[:, 1].max() + X_vals[:, 1].min()) * 0.5
    mid_z = (X_vals[:, 2].max() + X_vals[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Triangulated points (%d)' % Xf.shape[0])

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    base = os.path.splitext(out_path)[0]
    azs = np.linspace(0.0, 360.0, num=n_views, endpoint=False)
    for az in azs:
        ax.view_init(elev=view_elev, azim=az)
        view_file = "%s_az%03d_el%02d.png" % (base, int(round(az)) % 360, int(round(view_elev)))
        fig.savefig(view_file, dpi=200)
    ax.view_init(elev=view_elev, azim=float(azs[0]))
    fig.savefig(out_path, dpi=200)
    print('Saved %d 3D view images to %s* (including %s)' % (len(azs), base, out_path))


def main():
    parser = argparse.ArgumentParser(description='Triangulate using GT cameras (.camera files) and save 3D views')
    parser.add_argument('--images_dir', required=False,
                        default='Benchmarking_Camera_Calibration_2008/fountain-P11/images',
                        help='directory containing images (and optionally K.txt)')
    parser.add_argument('--gt_dir', required=False,
                        help='directory containing GT camera files (default: images_dir/gt_dense_cameras)')
    parser.add_argument('--images', required=False, nargs='+',
                        default=['0000.jpg', '0001.jpg', '0002.jpg'],
                        help='list of image filenames to use (space separated)')
    parser.add_argument('--out', required=False, default='outputs/triangulation_gt.png')
    parser.add_argument('--views', required=False, type=int, default=12,
                        help='number of azimuth views to save (rotating around scene)')
    parser.add_argument('--elev', required=False, type=float, default=20.0,
                        help='elevation angle for the saved views')
    parser.add_argument('--method', required=False, choices=['linear', 'nonlinear'], default='linear',
                        help='triangulation method to use (linear or nonlinear refine per-point)')
    parser.add_argument('--refine-iters', required=False, type=int, default=20,
                        help='number of LM iterations for nonlinear per-point refinement')
    args = parser.parse_args()

    # Resolve gt_dir: prefer explicit flag, otherwise try common locations next to images
    if args.gt_dir:
        gt_dir = args.gt_dir
    else:
        cand1 = os.path.join(args.images_dir, 'gt_dense_cameras')
        cand2 = os.path.join(os.path.dirname(args.images_dir), 'gt_dense_cameras')
        if os.path.exists(cand1):
            gt_dir = cand1
        elif os.path.exists(cand2):
            gt_dir = cand2
        else:
            gt_dir = cand1
            print(f'Warning: gt_dir not provided; tried {cand1} and {cand2} and none existed. Using {gt_dir} (may be incorrect)')
    print(f'Using gt_dir: {gt_dir}')
    triangulate_with_gt(args.images_dir, args.images, gt_dir, args.out,
                        method=args.method, refine_iters=args.refine_iters,
                        n_views=args.views, view_elev=args.elev)


if __name__ == '__main__':
    main()
