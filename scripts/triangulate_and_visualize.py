#!/usr/bin/env python3
"""
Multi-view triangulation and visualization utility.

Usage (example):
  python3 scripts/triangulate_and_visualize.py \
    --images_dir Benchmarking_Camera_Calibration_2008/fountain-P11/images \
    --images 0000.jpg 0001.jpg 0002.jpg --out outputs/triangulation.png

The script will:
 - read K.txt for the intrinsic matrix (expects file images_dir/K.txt)
 - detect features (SIFT preferred, falls back to ORB)
 - match descriptors pairwise and build tracks across images
 - estimate pairwise poses relative to the first image
 - triangulate tracks observed in multiple views using a linear multi-view DLT
 - render a 3D scatter plot of points and camera centres and save to the given output
 - save a concatenated image showing matches across all input images

This is a simple, single-file utility intended for quick visual checks.
"""
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_K(kfile):
    if not os.path.exists(kfile):
        raise FileNotFoundError("K matrix file not found: %s" % kfile)
    with open(kfile, 'r') as f:
        vals = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(';', ' ').split()
            for p in parts:
                try:
                    vals.append(float(p))
                except:
                    pass
        if len(vals) < 9:
            raise ValueError('K.txt does not contain 9 numeric values')
        K = np.array(vals[:9], dtype=float).reshape(3, 3)
        return K


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
    # ORB fallback
    orb = cv2.ORB_create(4000)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des, 'ORB'


def match_descriptors(des1, des2, method):
    # Returns list of cv2.DMatch objects (filtered by ratio test)
    if des1 is None or des2 is None:
        return []
    if method == 'SIFT':
        # FLANN
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
        self.parent = dict()

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        # path compression
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


def triangulate_and_plot_multi(images_dir, image_names, kfile, out_path,
                               method='linear', refine_iters=20,
                               n_views=8, view_elev=20.0):
    # image_names: ordered list of image filenames to use for multi-view triangulation
    if len(image_names) < 2:
        raise ValueError('Need at least two images')

    K = read_K(kfile)

    # Read images and detect features
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

    # Match descriptors pairwise and build tracks via union-find on (img_idx, kp_idx)
    uf = UnionFind()
    matches_by_pair = {}
    print('Matching descriptors between image pairs...')
    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            # only match if descriptor types are the same (SIFT or ORB). Otherwise skip
            if methods[i] != methods[j]:
                matches_by_pair[(i, j)] = []
                continue
            detector_method = methods[i]
            good = match_descriptors(dess[i], dess[j], detector_method)
            matches_by_pair[(i, j)] = good
            # Register in union-find
            for m in good:
                a = (i, m.queryIdx)
                b = (j, m.trainIdx)
                uf.add(a)
                uf.add(b)
                uf.union(a, b)

    # Build tracks: each connected component corresponds to a 2D track across images
    tracks = {}
    for key in uf.parent.keys():
        root = uf.find(key)
        tracks.setdefault(root, set()).add(key)

    # Convert tracks to lists and filter tracks observed in at least two images
    good_tracks = []
    for tr in tracks.values():
        obs = list(tr)
        imgs_seen = set([i for (i, k) in obs])
        if len(imgs_seen) >= 2:
            good_tracks.append(obs)

    print('Built %d tracks (%d with >=2 views)' % (len(tracks), len(good_tracks)))

    # For each track, compute the best triangulated 3D point using linear method over all observations
    # We'll first estimate relative camera poses by chaining pairwise poses from the first image.
    # For simplicity, compute poses relative to image 0 via pairwise essential+recoverPose to image0.
    poses_R = [None] * len(image_names)
    poses_t = [None] * len(image_names)
    poses_R[0] = np.eye(3)
    poses_t[0] = np.zeros((3, 1))

    # compute pairwise pose relative to image 0 when possible
    for j in range(1, len(image_names)):
        # match between 0 and j
        good = matches_by_pair.get((0, j), [])
        if len(good) < 8:
            good = matches_by_pair.get((j, 0), [])
            # need to swap indices
            swapped = True
        else:
            swapped = False
        if len(good) < 8:
            print('Warning: not enough matches between image 0 and', j)
            continue

        # build point arrays
        if not swapped:
            pts1 = np.array([kps[0][m.queryIdx].pt for m in good], dtype=float)
            pts2 = np.array([kps[j][m.trainIdx].pt for m in good], dtype=float)
        else:
            pts1 = np.array([kps[0][m.trainIdx].pt for m in good], dtype=float)
            pts2 = np.array([kps[j][m.queryIdx].pt for m in good], dtype=float)

        E, maskE = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print('Warning: essential matrix failed between 0 and', j)
            continue
        _, R, t, maskPose = cv2.recoverPose(E, pts1, pts2, K)
        poses_R[j] = R
        poses_t[j] = t

    # Triangulate tracks that have known poses for their observing images
    points3d = []
    tracks_used = []
    for tr in good_tracks:
        obs = list(tr)
        # For each observation, check if we know the pose
        A_obs = []
        bs = []
        img_indices = []
        Pi_list_for_track = []
        uvs_for_track = []
        for (i, kp_idx) in obs:
            if poses_R[i] is None or poses_t[i] is None:
                continue
            pt = kps[i][kp_idx].pt
            x, y = pt
            # build projection matrix Pi = K [R|t]
            Ri = poses_R[i]
            ti = poses_t[i]
            Pi = K.dot(np.hstack((Ri, ti)))
            A_obs.append(Pi)
            bs.append((i, x, y))
            img_indices.append(i)
            Pi_list_for_track.append(Pi)
            uvs_for_track.append((x, y))

        if len(A_obs) < 2:
            continue

        # Linear triangulation using all views (DLT)
        # Build linear system: for each view, x * (P[2,:] X) - P[0,:] X = 0, y * (P[2,:] X) - P[1,:] X = 0
        M = []
        for Pi, (_, x, y) in zip(A_obs, bs):
            P0 = Pi[0, :]
            P1 = Pi[1, :]
            P2 = Pi[2, :]
            M.append(x * P2 - P0)
            M.append(y * P2 - P1)
        M = np.array(M)
        # Solve by SVD
        try:
            _, _, Vt = np.linalg.svd(M)
        except np.linalg.LinAlgError:
            continue
        X = Vt[-1, :]
        if abs(X[-1]) < 1e-8:
            continue
        X = X[:3] / X[3]

        points3d.append(X)
        tracks_used.append({'obs': obs, 'img_indices': img_indices, 'Pi_list': Pi_list_for_track, 'uvs': uvs_for_track})

    if len(points3d) == 0:
        raise RuntimeError('No 3D points reconstructed')

    Xf = np.array(points3d)
    print('Triangulated %d points from %d tracks' % (Xf.shape[0], len(tracks_used)))

    #----------------------------------------------------------------------------------
    # Reconstruction error reporting and optional non-linear refinement per point
    #----------------------------------------------------------------------------------
    def reprojection_errors_point(X, Pi_list, uvs):
        # returns per-observation reprojection errors (pixels)
        errs = []
        for Pi, (u_obs, v_obs) in zip(Pi_list, uvs):
            P0 = Pi[0, :3]
            p04 = Pi[0, 3]
            P1 = Pi[1, :3]
            p14 = Pi[1, 3]
            P2 = Pi[2, :3]
            p24 = Pi[2, 3]
            N_u = float(P0.dot(X) + p04)
            N_v = float(P1.dot(X) + p14)
            D = float(P2.dot(X) + p24)
            if abs(D) < 1e-12:
                # put large error
                errs.append(1e6)
                continue
            u_proj = N_u / D
            v_proj = N_v / D
            e = np.hypot(u_proj - u_obs, v_proj - v_obs)
            errs.append(e)
        return np.array(errs, dtype=float)

    def refine_point_lm(X0, Pi_list, uvs, max_iter=20, tol=1e-6, verbose=False):
        # Levenberg-Marquardt style iterative refinement on 3 parameters (X, Y, Z)
        X = np.array(X0, dtype=float).reshape(3)
        # initial cost
        errs = reprojection_errors_point(X, Pi_list, uvs)
        cost = float((errs ** 2).sum())
        lm_lambda = 1e-3
        for it in range(max_iter):
            # build Jacobian and residuals
            J_rows = []
            r = []
            bad = False
            for Pi, (u_obs, v_obs) in zip(Pi_list, uvs):
                n0 = Pi[0, :3]
                n0_4 = Pi[0, 3]
                n1 = Pi[1, :3]
                n1_4 = Pi[1, 3]
                n2 = Pi[2, :3]
                n2_4 = Pi[2, 3]
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
                # Jacobians: du/dX = (n0*D - Nu*n2) / D^2
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
            # normal equations
            H = J.T.dot(J)
            g = J.T.dot(r)
            # augment diagonal
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
                # accept and decrease lambda
                X = X_new
                cost = cost_new
                lm_lambda = max(lm_lambda * 0.1, 1e-12)
            else:
                lm_lambda *= 10.0
        final_errs = reprojection_errors_point(X, Pi_list, uvs)
        return X, final_errs

    # compute reprojection errors for all points (before/after refinement)
    reproj_before = []
    reproj_after = []
    refined_points = []
    for X, info in zip(points3d, tracks_used):
        # info is a dict with keys 'obs','img_indices','Pi_list','uvs' (or older tuple form)
        if isinstance(info, dict):
            obs = info.get('obs', [])
            Pi_list = info.get('Pi_list', [])
            uvs = info.get('uvs', [])
            # construct Pi_list/uvs if missing
            if not Pi_list:
                Pi_list = []
                uvs = []
                for (i, kp_idx) in obs:
                    if poses_R[i] is None or poses_t[i] is None:
                        continue
                    Pi = K.dot(np.hstack((poses_R[i], poses_t[i])))
                    Pi_list.append(Pi)
                    u, v = kps[i][kp_idx].pt
                    uvs.append((u, v))
        else:
            # older tuple form
            obs, img_indices = info
            Pi_list = []
            uvs = []
            for (i, kp_idx) in obs:
                if poses_R[i] is None or poses_t[i] is None:
                    continue
                Pi = K.dot(np.hstack((poses_R[i], poses_t[i])))
                Pi_list.append(Pi)
                u, v = kps[i][kp_idx].pt
                uvs.append((u, v))
        if len(Pi_list) < 2:
            continue
        errs0 = reprojection_errors_point(X, Pi_list, uvs)
        reproj_before.append(errs0.mean())
        if method == 'nonlinear':
            X_refined, errs_ref = refine_point_lm(X, Pi_list, uvs, max_iter=refine_iters)
            refined_points.append(X_refined)
            reproj_after.append(errs_ref.mean())
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

    # replace Xf with refined points for visualization
    Xf = np.array(refined_points)

    # Visualize matches across all image pairs used for track building
    try:
        # Build a horizontal concatenation visualization for all images
        vis_rows = []
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

        # draw matches for each pair, in gray lines; tracks used for reconstruction in green
        for (i, j), matches in matches_by_pair.items():
            for m in matches:
                x1, y1 = kps[i][m.queryIdx].pt
                x2, y2 = kps[j][m.trainIdx].pt
                p1 = (int(round(x_offsets[i] + x1)), int(round(y1)))
                p2 = (int(round(x_offsets[j] + x2)), int(round(y2)))
                cv2.line(vis, p1, p2, (150, 150, 150), 1, lineType=cv2.LINE_AA)

    # highlight tracks used in reconstruction
        for tr_idx, info in enumerate(tracks_used):
            # info may be a dict or tuple
            if isinstance(info, dict):
                obs = info.get('obs', [])
            else:
                obs, _ = info
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))
            for (i, kp_idx) in obs:
                x, y = kps[i][kp_idx].pt
                p = (int(round(x_offsets[i] + x)), int(round(y)))
                cv2.circle(vis, p, 3, color, -1, lineType=cv2.LINE_AA)

        matches_out = os.path.splitext(out_path)[0] + '_matches_multi.png'
        cv2.imwrite(matches_out, vis)
        print('Saved multi-image matches visualization to %s' % matches_out)
    except Exception as e:
        print('Warning: failed to save multi-image matches visualization: %s' % str(e))

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xf[:, 0], Xf[:, 1], Xf[:, 2], s=1, c='b', alpha=0.8)

    # plot camera centers for all images
    cam_centers = []
    for i in range(len(image_names)):
        if poses_R[i] is None or poses_t[i] is None:
            continue
        R = poses_R[i]
        t = poses_t[i]
        C = -R.T.dot(t).ravel()
        cam_centers.append(C)
        ax.scatter(C[0], C[1], C[2], s=50, marker='^')
        ax.text(C[0], C[1], C[2], 'Cam%d' % i)

    # also show origin camera 0
    ax.scatter(0, 0, 0, c='r', s=60, marker='x')

    # set equal aspect
    if Xf.size == 0:
        X_vals = np.zeros((1, 3))
    else:
        X_vals = Xf
    max_range = np.array([X_vals[:, 0].max() - X_vals[:, 0].min(),
                          X_vals[:, 1].max() - X_vals[:, 1].min(),
                          X_vals[:, 2].max() - X_vals[:, 2].min()]).max() / 2.0
    mid_x = (X_vals[:, 0].max() + X_vals[:, 0].min()) * 0.5
    mid_y = (X_vals[:, 1].max() + X_vals[:, 1].min()) * 0.5
    mid_z = (X_vals[:, 2].max() + X_vals[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triangulated points (%d)' % Xf.shape[0])

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save multiple views by rotating azimuth around the scene
    base = os.path.splitext(out_path)[0]
    azs = np.linspace(0.0, 360.0, num=n_views, endpoint=False)
    for az in azs:
        ax.view_init(elev=view_elev, azim=az)
        view_file = "%s_az%03d_el%02d.png" % (base, int(round(az)) % 360, int(round(view_elev)))
        fig.savefig(view_file, dpi=200)
    # also save the provided out_path for convenience (first az)
    ax.view_init(elev=view_elev, azim=float(azs[0]))
    fig.savefig(out_path, dpi=200)
    print('Saved %d 3D view images to %s* (including %s)' % (len(azs), base, out_path))


def main():
    parser = argparse.ArgumentParser(description='Triangulate points between multiple images and save 3D view')
    parser.add_argument('--images_dir', required=False,
                        default='Benchmarking_Camera_Calibration_2008/fountain-P11/images',
                        help='directory containing images and K.txt')
    parser.add_argument('--images', required=False, nargs='+',
                        default=['0000.jpg', '0001.jpg', '0002.jpg'],
                        help='list of image filenames to use (space separated)')
    parser.add_argument('--K', required=False, help='K matrix file (default: images_dir/K.txt)')
    parser.add_argument('--out', required=False, default='outputs/triangulation.png')
    parser.add_argument('--views', required=False, type=int, default=8,
                        help='number of azimuth views to save (rotating around scene)')
    parser.add_argument('--elev', required=False, type=float, default=20.0,
                        help='elevation angle for the saved views')
    parser.add_argument('--method', required=False, choices=['linear', 'nonlinear'], default='linear',
                        help='triangulation method to use (linear or nonlinear refine per-point)')
    parser.add_argument('--refine-iters', required=False, type=int, default=20,
                        help='number of LM iterations for nonlinear per-point refinement')
    args = parser.parse_args()

    kfile = args.K if args.K else os.path.join(args.images_dir, 'K.txt')
    triangulate_and_plot_multi(args.images_dir, args.images, kfile, args.out,
                               method=args.method, refine_iters=args.refine_iters,
                               n_views=args.views, view_elev=args.elev)


if __name__ == '__main__':
    main()
