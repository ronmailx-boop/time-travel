"""Find where each video frame sits inside the stitched panorama -> per-frame view direction.
Usage: python scripts/track_yaw.py <video> <pano.jpg> <out.json> [focal_px_at_720=780]
Output: {"fps": n, "f": f, "pano": [w,h], "track": [[t, cx, cy, roll_deg, scale], ...]} in pano pixels.
"""
import json, sys
import cv2, numpy as np

video, pano_p, out = sys.argv[1:4]
f = float(sys.argv[4]) if len(sys.argv) > 4 else 780.0
pano = cv2.imread(pano_p, cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create(4000)
kp_p, des_p = sift.detectAndCompute(pano, None)
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(round(fps / 6)))  # 6 samples per second
track, i = [], 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i % step == 0:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (720, int(round(g.shape[0] * 720 / g.shape[1]))))
        kp, des = sift.detectAndCompute(g, None)
        if des is not None and len(kp) > 20:
            m = [a for a, b in matcher.knnMatch(des, des_p, k=2) if a.distance < 0.7 * b.distance]
            if len(m) >= 25:
                src = np.float32([kp[x.queryIdx].pt for x in m]); dst = np.float32([kp_p[x.trainIdx].pt for x in m])
                H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
                if H is not None and inl.sum() >= 20:
                    h, w = g.shape
                    pts = cv2.perspectiveTransform(np.float32([[[w/2, h/2]], [[w/2, 0]], [[w/2, h]]]), H)[:, 0]
                    c, top, bot = pts
                    v = bot - top
                    roll = float(np.degrees(np.arctan2(-v[0], v[1])))
                    scale = float(np.linalg.norm(v) / h)
                    track.append([round(i / fps, 3), round(float(c[0]), 1), round(float(c[1]), 1), round(roll, 2), round(scale, 3)])
    i += 1
json.dump({"fps": fps, "f": f, "pano": [pano.shape[1], pano.shape[0]], "track": track}, open(out, "w"))
print(len(track), "matched of", i // step + 1)
