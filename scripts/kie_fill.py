"""Fill the missing parts of the 360 with Nano Banana Edit on kie.ai (4 calls, ~$0.02 each).
Usage: python scripts/kie_fill.py <pano.jpg> <out_dir> [--run]   (without --run it only writes the crops)
Real pixels are always put back; the model only fills what the video never saw.
"""
import base64, json, math, os, sys, time, urllib.request
import cv2, numpy as np

pano_p, out = sys.argv[1], sys.argv[2]
RUN = "--run" in sys.argv
F, EW = 780.0, 4096
EH = EW // 2
os.makedirs(out, exist_ok=True)

# --- real panorama on the sphere (same placement as place_equirect.py)
pano = cv2.imread(pano_p); ph, pw = pano.shape[:2]
nw = int(round(EW * (pw / F) / (2 * math.pi))); nh = int(round(EH * (ph / F) / math.pi))
small = cv2.resize(pano, (nw, nh), interpolation=cv2.INTER_AREA)
valid = cv2.erode((cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) > 3).astype(np.uint8), np.ones((5, 5), np.uint8))
x0, y0 = (EW - nw) // 2, (EH - nh) // 2
eq = np.zeros((EH, EW, 3), np.uint8); known = np.zeros((EH, EW), np.uint8)
eq[y0:y0+nh, x0:x0+nw][valid > 0] = small[valid > 0]; known[y0:y0+nh, x0:x0+nw] = valid
real = known.copy()

def edge_ramp(h, w, r=320):
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]; xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    return np.clip(np.minimum(yy, xx) / r, 0, 1).astype(np.float32)

def merge(base, base_known, base_real, new, ramp):
    """Blend a generated patch into existing content: real pixels stay, earlier fills fade into the new one."""
    w = np.where(base_known > 0, ramp, 1.0)
    w = np.where(base_real > 0, 0.0, w)[..., None]
    out = (base * (1 - w) + new * w).astype(np.uint8)
    newk = np.maximum(base_known, (new.max(axis=2) > 12).astype(np.uint8))
    return out, newk

PROMPT = ("This is part of a 360-degree photo taken on a sunny summer afternoon from a high mountain viewing platform "
          "above an alpine lake (concrete platform floor, metal railing with wire mesh, forested limestone mountains, "
          "blue sky with a few white clouds, some tourists). Fill ONLY the pure black empty areas so the photo becomes "
          "complete and continuous. Keep every existing (non-black) pixel exactly as it is, same perspective, same light, "
          "same colours, same camera quality. No borders, no frames, no text. Photorealistic smartphone photo.")
PROMPT_UP = PROMPT + " This view looks straight up at the sky: fill it with the same blue sky and soft clouds."
PROMPT_DOWN = PROMPT + " This view looks straight down at the platform floor: fill it with the same grey concrete floor."

REUSE = "--reuse" in sys.argv
def kie(path, prompt, ratio):
    cached = path.replace(".jpg", "_out.png")
    if REUSE and os.path.exists(cached):
        return cached
    data = "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()
    def post(url, body):
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer proxy-injected"})
        return json.load(urllib.request.urlopen(req, timeout=120))
    up = post("https://kieai.redpandaai.co/api/file-base64-upload", {"base64Data": data, "uploadPath": "time-travel", "fileName": os.path.basename(path)})
    url = up["data"]["downloadUrl"]
    task = post("https://api.kie.ai/api/v1/jobs/createTask", {"model": "google/nano-banana-edit",
            "input": {"prompt": prompt, "image_urls": [url], "output_format": "png", "aspect_ratio": ratio}})
    tid = task["data"]["taskId"]; print("task", tid, flush=True)
    for _ in range(120):
        time.sleep(5)
        req = urllib.request.Request(f"https://api.kie.ai/api/v1/jobs/recordInfo?taskId={tid}", headers={"Authorization": "Bearer proxy-injected"})
        d = json.load(urllib.request.urlopen(req, timeout=60))["data"]
        if d["state"] == "success":
            res = json.loads(d["resultJson"])["resultUrls"][0]
            dst = path.replace(".jpg", "_out.png"); urllib.request.urlretrieve(res, dst); return dst
        if d["state"] == "fail":
            raise RuntimeError(d.get("failMsg"))
    raise TimeoutError(tid)

def paste(crop_bgr, crop_known, gen_path, feather=41, keep=None):
    g = cv2.resize(cv2.imread(gen_path), (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    # match the generated colours to the real ones on the known area (simple per-channel mean/std transfer)
    m = crop_known > 0
    if m.sum() > 1000:
        for c in range(3):
            gs, rs = g[..., c][m].astype(np.float32), crop_bgr[..., c][m].astype(np.float32)
            g[..., c] = np.clip((g[..., c] - gs.mean()) / (gs.std() + 1e-3) * rs.std() + rs.mean(), 0, 255)
    keep = crop_known if keep is None else keep
    k = cv2.GaussianBlur(cv2.erode(keep, np.ones((9, 9), np.uint8)).astype(np.float32), (feather, feather), 0)[..., None]
    return (crop_bgr * k + g * (1 - k)).astype(np.uint8)

# --- 1+2) the band (lat +-55deg) in two overlapping halves: back (centred on 180deg) and front (centred on 0)
lat_top = int(EH * (0.5 - 55 / 180)); lat_bot = int(EH * (0.5 + 55 / 180))
half_w = int(EW * 215 / 360)  # 215deg wide -> overlap with the other half
jobs = [("front", EW // 2), ("back", 0)]
for name, centre in jobs:
    eq_r = np.roll(eq, EW // 2 - centre, axis=1); kn_r = np.roll(known, EW // 2 - centre, axis=1); re_r = np.roll(real, EW // 2 - centre, axis=1)
    xa = EW // 2 - half_w // 2
    crop = eq_r[lat_top:lat_bot, xa:xa + half_w].copy(); ck = kn_r[lat_top:lat_bot, xa:xa + half_w].copy()
    p = f"{out}/band_{name}.jpg"; cv2.imwrite(p, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(name, crop.shape, "known %.0f%%" % (100 * ck.mean()))
    if RUN:
        rc = re_r[lat_top:lat_bot, xa:xa + half_w]
        filled = paste(crop, ck, kie(p, PROMPT, "16:9"), keep=rc)
        sl = (slice(lat_top, lat_bot), slice(xa, xa + half_w))
        eq_r[sl], kn_r[sl] = merge(eq_r[sl], kn_r[sl], re_r[sl], filled, edge_ramp(*filled.shape[:2]))
        eq = np.roll(eq_r, -(EW // 2 - centre), axis=1); known = np.roll(kn_r, -(EW // 2 - centre), axis=1)
        cv2.imwrite(f"{out}/eq_after_{name}.jpg", eq, [cv2.IMWRITE_JPEG_QUALITY, 88])

# --- 3+4) sky and floor through cube faces
S = 1024
def face_maps(face):
    a = (np.arange(S) + 0.5) / S * 2 - 1; u, v = np.meshgrid(a, a)
    x, y, z = (u, np.ones_like(u), v) if face == "top" else (u, -np.ones_like(u), -v)
    lon = np.arctan2(x, -z); lat = np.arctan2(y, np.sqrt(x * x + z * z))
    return ((lon / (2 * np.pi) + 0.5) * EW).astype(np.float32), ((0.5 - lat / np.pi) * EH).astype(np.float32)
for face, prompt in (("top", PROMPT_UP), ("bottom", PROMPT_DOWN)):
    mx, my = face_maps(face)
    fimg = cv2.remap(eq, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    fk = (cv2.remap(known * 255, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 127).astype(np.uint8)
    p = f"{out}/face_{face}.jpg"; cv2.imwrite(p, fimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(face, "known %.0f%%" % (100 * fk.mean()))
    if RUN:
        fimg = paste(fimg, fk, kie(p, prompt, "1:1"))
        lon = (np.arange(EW) + 0.5) / EW * 2 * np.pi - np.pi; lat = np.pi / 2 - (np.arange(EH) + 0.5) / EH * np.pi
        LON, LAT = np.meshgrid(lon, lat)
        x = np.cos(LAT) * np.sin(LON); y = np.sin(LAT); z = -np.cos(LAT) * np.cos(LON)
        if face == "top":
            sel = y > np.maximum(np.abs(x), np.abs(z)); d = np.where(sel, y, 1); u, v = x / d, z / d
        else:
            sel = -y > np.maximum(np.abs(x), np.abs(z)); d = np.where(sel, -y, 1); u, v = x / d, -z / d
        fx = ((u + 1) / 2 * S - 0.5).astype(np.float32); fy = ((v + 1) / 2 * S - 0.5).astype(np.float32)
        samp = cv2.remap(fimg, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cheb = np.maximum(np.abs(u), np.abs(v))
        w = np.where(sel, np.clip((1 - cheb) / 0.12, 0, 1), 0)          # fade out towards the face edge
        if face == "top":   # the sky cap only near the zenith; lower down keep the band (avoids floating mountains)
            w = w * np.clip((np.abs(LAT) - np.radians(60)) / np.radians(12), 0, 1)
        w = np.where(known == 0, sel.astype(np.float64), w)              # holes: take the face fully
        w = np.where(real > 0, 0, w)[..., None]
        eq = (eq * (1 - w) + samp * w).astype(np.uint8); known = np.maximum(known, (sel & (samp.max(axis=2) > 12)).astype(np.uint8))

if RUN:
    rest = (known == 0).astype(np.uint8)
    if rest.any(): eq = cv2.inpaint(eq, rest, 7, cv2.INPAINT_TELEA)
    # Clean sky cap: above ~52deg use a smooth gradient from the band's top edge to a zenith blue (no invented objects)
    r0 = int(EH * (0.5 - 52 / 180)); edge = cv2.GaussianBlur(eq[r0:r0 + 6].mean(axis=0, keepdims=True).astype(np.float32), (0, 0), 60)
    zen = np.median(eq[r0:r0 + 6].reshape(-1, 3), axis=0).astype(np.float32) * np.array([1.0, 0.92, 0.85], np.float32)
    for y in range(r0):
        t = (r0 - y) / r0; t = t * t * (3 - 2 * t)
        row = edge[0] * (1 - t) + zen * t
        a = np.clip((r0 - y) / 40.0, 0, 1)
        keep = real[y] > 0
        eq[y] = np.where(keep[:, None], eq[y], (eq[y] * (1 - a) + row * a)).astype(np.uint8)
    cv2.imwrite(f"{out}/equirect_ai.jpg", eq, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("done")
