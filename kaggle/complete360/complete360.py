# Step 1: turn the real (stitched) panorama into a full 360x180 equirectangular image.
# Real pixels are kept; only the empty areas (behind, sky, floor) are filled with SDXL inpainting.
# Runs on Kaggle GPU. Input: output of vplusstudio/time-travel-stitch (src.mp4 + panorama).
import glob, json, math, os, subprocess, sys, time
import cv2, numpy as np

OUT = "/kaggle/working"
IN = glob.glob("/kaggle/input/**/src.mp4", recursive=True)
assert IN, "stitch output not mounted"
SRC = IN[0]
print("src", SRC, flush=True)
info = {}

# 1) Re-stitch to read the camera focal length (needed to map the panorama onto the sphere)
FR = "/kaggle/tmp/fr"; os.makedirs(FR, exist_ok=True)
subprocess.run(f"ffmpeg -hide_banner -loglevel error -i {SRC} -vf fps=2,scale=720:-2 -q:v 2 {FR}/f_%03d.jpg", shell=True, check=True)
imgs = [cv2.imread(f) for f in sorted(glob.glob(f"{FR}/*.jpg"))]
st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
st.setPanoConfidenceThresh(0.6)
status, pano = st.stitch(imgs)
assert status == cv2.Stitcher_OK, status
focals = []
try:
    focals = [float(c.focal) for c in st.cameras()]
except Exception as e:
    print("cameras() unavailable:", e)
f_med = float(np.median(focals)) if focals else 0.0
# Phone main camera (~26mm equiv.) in portrait: short side ~50 deg -> f ~= 360/tan(25deg) ~= 780 px at width 720
f = f_med if 400 < f_med < 2000 else 780.0
info.update(pano_hw=list(pano.shape[:2]), focal_median=f_med, focal_used=f)
print(info, flush=True)

# 2) Place the spherical panorama into an equirectangular canvas (360 x 180 deg)
EW, EH = 4096, 2048
ph, pw = pano.shape[:2]
span_x = pw / f            # radians
span_y = ph / f
info.update(hfov_deg=math.degrees(span_x), vfov_deg=math.degrees(span_y))
nw = int(round(EW * span_x / (2 * math.pi)))
nh = int(round(EH * span_y / math.pi))
small = cv2.resize(pano, (nw, nh), interpolation=cv2.INTER_AREA)
valid = (cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) > 3).astype(np.uint8)
valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
x0 = (EW - nw) // 2
y0 = (EH - nh) // 2
eq = np.zeros((EH, EW, 3), np.uint8)
known = np.zeros((EH, EW), np.uint8)
eq[y0:y0 + nh, x0:x0 + nw][valid > 0] = small[valid > 0]
known[y0:y0 + nh, x0:x0 + nw] = valid
cv2.imwrite(f"{OUT}/eq_real.jpg", eq, [cv2.IMWRITE_JPEG_QUALITY, 90])
info["placement"] = dict(x0=x0, y0=y0, w=nw, h=nh)

# 3) Inpaint the missing areas with SDXL. Work on a rolled copy so the empty "behind" part is in the middle.
subprocess.run(f"{sys.executable} -m pip install -q -U diffusers accelerate", shell=True, check=True)
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")
PROMPT = ("360 degree photo from a summer mountain viewing platform high above an alpine lake, "
          "steep forested limestone mountains, clear blue sky with small white clouds, concrete platform floor, "
          "metal railing with wire mesh, green trees, tourists, smartphone photo, natural light, realistic")
NEG = "painting, illustration, cartoon, text, watermark, frame, border, blurry, distorted, seams"

def inpaint(img_bgr, mask, steps=30, strength=0.99, seed=7):
    h, w = img_bgr.shape[:2]
    W, H = (2048, 1024) if w >= h * 1.5 else (1024, 1024)
    im = Image.fromarray(cv2.cvtColor(cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    mk = Image.fromarray(cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) * 255)
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe(prompt=PROMPT, negative_prompt=NEG, image=im, mask_image=mk, width=W, height=H,
               num_inference_steps=steps, strength=strength, guidance_scale=7.0, generator=g).images[0]
    return cv2.resize(cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_CUBIC)

def blend(base, gen, known_mask, feather=31):
    k = cv2.GaussianBlur(known_mask.astype(np.float32), (feather, feather), 0)[..., None]
    return (base * k + gen * (1 - k)).astype(np.uint8)

t = time.time()
# 3a) Horizontal band (behind + sides): roll so the gap is centred
shift = EW // 2
eq_r = np.roll(eq, shift, axis=1); kn_r = np.roll(known, shift, axis=1)
band_y0, band_y1 = EH // 8, EH - EH // 8          # the band between ~67deg up and ~67deg down
band = eq_r[band_y0:band_y1]; bk = kn_r[band_y0:band_y1]
# Coarse fill first so the model sees colours instead of black, then generate
coarse = cv2.inpaint(band, (1 - bk).astype(np.uint8), 5, cv2.INPAINT_TELEA)
gen = inpaint(coarse, (1 - bk).astype(np.uint8))
band_f = blend(band, gen, bk)
eq_r[band_y0:band_y1] = band_f; kn_r[band_y0:band_y1] = 1
eq = np.roll(eq_r, -shift, axis=1); known = np.roll(kn_r, -shift, axis=1)
cv2.imwrite(f"{OUT}/eq_band.jpg", eq, [cv2.IMWRITE_JPEG_QUALITY, 90])

# 3b) Poles via cube faces (top = sky, bottom = floor): sample face, inpaint, write back
def face_maps(face, size):
    a = (np.arange(size) + 0.5) / size * 2 - 1
    u, v = np.meshgrid(a, a)
    if face == "top":    x, y, z = u, np.ones_like(u), v
    else:                x, y, z = u, -np.ones_like(u), -v
    lon = np.arctan2(x, z); lat = np.arctan2(y, np.sqrt(x * x + z * z))
    mx = ((lon / (2 * np.pi) + 0.5) * EW).astype(np.float32)
    my = ((0.5 - lat / np.pi) * EH).astype(np.float32)
    return mx, my, lon, lat

S = 1024
for face in ["top", "bottom"]:
    mx, my, _, _ = face_maps(face, S)
    fimg = cv2.remap(eq, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    fk = cv2.remap(known * 255, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) // 255
    if fk.mean() > 0.98:
        continue
    coarse = cv2.inpaint(fimg, (1 - fk).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    gen = inpaint(coarse, (1 - fk).astype(np.uint8), seed=11 if face == "top" else 13)
    fimg = blend(fimg, gen, fk)
    cv2.imwrite(f"{OUT}/face_{face}.jpg", fimg)
    # write back: for each equirect pixel near this pole, look up the face pixel
    lon = (np.arange(EW) + 0.5) / EW * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(EH) + 0.5) / EH * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    x = np.cos(LAT) * np.sin(LON); y = np.sin(LAT); z = np.cos(LAT) * np.cos(LON)
    if face == "top":
        sel = y > np.maximum(np.abs(x), np.abs(z)); u = x / np.where(sel, y, 1); v = z / np.where(sel, y, 1)
    else:
        sel = -y > np.maximum(np.abs(x), np.abs(z)); u = x / np.where(sel, -y, 1); v = -z / np.where(sel, -y, 1)
    fx = ((u + 1) / 2 * S - 0.5).astype(np.float32); fy = ((v + 1) / 2 * S - 0.5).astype(np.float32)
    samp = cv2.remap(fimg, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m = sel & (known == 0)
    eq[m] = samp[m]; known[m] = 1

# Anything still empty (cube-face corners): quick classical fill
rest = (known == 0).astype(np.uint8)
if rest.any():
    eq = cv2.inpaint(eq, rest, 5, cv2.INPAINT_TELEA)
info["gen_sec"] = round(time.time() - t, 1)
cv2.imwrite(f"{OUT}/equirect_4096.jpg", eq, [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(f"{OUT}/equirect_preview.jpg", cv2.resize(eq, (2048, 1024), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(info, open(f"{OUT}/info.json", "w"), indent=1)
print("done", info, flush=True)
