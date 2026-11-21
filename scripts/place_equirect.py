"""Place the stitched spherical panorama into a 360x180 equirectangular canvas (no AI).
Usage: python scripts/place_equirect.py <pano.jpg> <track.json> <out_dir> [focal=780] [width=4096]
Writes equirect_real.jpg (empty areas filled with a soft blur so it is not pure black) and track_lonlat.json.
"""
import json, math, sys
import cv2, numpy as np

pano_p, track_p, out = sys.argv[1:4]
f = float(sys.argv[4]) if len(sys.argv) > 4 else 780.0
EW = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
EH = EW // 2
pano = cv2.imread(pano_p)
ph, pw = pano.shape[:2]
nw = int(round(EW * (pw / f) / (2 * math.pi))); nh = int(round(EH * (ph / f) / math.pi))
small = cv2.resize(pano, (nw, nh), interpolation=cv2.INTER_AREA)
valid = (cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) > 3).astype(np.uint8)
valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
x0, y0 = (EW - nw) // 2, (EH - nh) // 2
eq = np.zeros((EH, EW, 3), np.uint8); known = np.zeros((EH, EW), np.uint8)
eq[y0:y0+nh, x0:x0+nw][valid > 0] = small[valid > 0]; known[y0:y0+nh, x0:x0+nw] = valid
# temporary fill for unknown areas: heavy blur pulled outward (placeholder until the AI fill)
fill = eq.copy().astype(np.float32); w = known.astype(np.float32)
for k in (31, 101, 301, 801):
    b = cv2.blur(eq.astype(np.float32) * w[..., None], (k, k)); bw = cv2.blur(w, (k, k))[..., None]
    est = b / np.maximum(bw, 1e-4)
    m = (w == 0) & (bw[..., 0] > 1e-3)
    fill[m] = est[m]; w = np.where(m, 1.0, w)
fill[w == 0] = (90, 70, 50)
out_img = np.where(known[..., None] > 0, eq, (fill * 0.75).astype(np.uint8))
cv2.imwrite(f"{out}/equirect_real.jpg", out_img, [cv2.IMWRITE_JPEG_QUALITY, 88])

# Convert the per-frame track (pano pixels) to view directions (radians), drop outliers, smooth
d = json.load(open(track_p))
rows = [r for r in d["track"] if 0.5 <= r[4] <= 2.2 and abs(r[3]) < 15]
t = np.array([r[0] for r in rows]); lon = np.array([(r[1] - pw / 2) / f for r in rows])
lat = np.array([(ph / 2 - r[2]) / f for r in rows]); roll = np.radians([r[3] for r in rows])
sc = np.array([r[4] for r in rows])
def smooth(a, k=5):
    p = np.pad(a, (k // 2, k // 2), mode="edge"); return np.convolve(p, np.ones(k) / k, mode="valid")
res = {"f": f, "frame_w": 720, "frame_h": 1280,
       "t": t.round(3).tolist(), "lon": smooth(lon).round(4).tolist(), "lat": smooth(lat).round(4).tolist(),
       "roll": smooth(roll).round(4).tolist(), "scale": smooth(sc).round(3).tolist()}
json.dump(res, open(f"{out}/track_lonlat.json", "w"))
print("kept", len(rows), "of", len(d["track"]), "lon range deg", np.degrees(lon.min()).round(1), np.degrees(lon.max()).round(1))
