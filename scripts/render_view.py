"""Render a normal (perspective) photo from the 360 equirect image.
Usage: python scripts/render_view.py <equirect.jpg> <out.jpg> [yaw_deg=0] [pitch_deg=0] [hfov_deg=90] [w=1280] [h=720]"""
import sys, math
import cv2, numpy as np
src, out = sys.argv[1:3]
yaw, pitch, hfov = (float(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((3, 0), (4, 0), (5, 90)))
W = int(sys.argv[6]) if len(sys.argv) > 6 else 1280; H = int(sys.argv[7]) if len(sys.argv) > 7 else 720
eq = cv2.imread(src); EH, EW = eq.shape[:2]
f = (W / 2) / math.tan(math.radians(hfov) / 2)
x, y = np.meshgrid(np.arange(W) - W / 2 + 0.5, np.arange(H) - H / 2 + 0.5)
d = np.stack([x, -y, -np.full_like(x, f)], -1); d /= np.linalg.norm(d, axis=-1, keepdims=True)
p, w = math.radians(pitch), math.radians(yaw)
Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
Ry = np.array([[math.cos(w), 0, -math.sin(w)], [0, 1, 0], [math.sin(w), 0, math.cos(w)]])
d = d @ (Ry @ Rx).T
lon = np.arctan2(d[..., 0], -d[..., 2]); lat = np.arcsin(np.clip(d[..., 1], -1, 1))
mx = ((lon / (2 * math.pi) + 0.5) * EW).astype(np.float32); my = ((0.5 - lat / math.pi) * EH).astype(np.float32)
cv2.imwrite(out, cv2.remap(eq, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP), [cv2.IMWRITE_JPEG_QUALITY, 93])
