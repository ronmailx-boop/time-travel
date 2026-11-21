# Step 0b: stitch a real panorama from the panning video frames (OpenCV, CPU).
# Also keeps a private copy of the source video in the output, for later notebooks.
import glob, json, os, shutil, subprocess, sys, time
import cv2

FILE_ID = "18ayaR52j9BU0OVed-_8JfyIyi8ucq8BR"
OUT = "/kaggle/working"
SRC = f"{OUT}/src.mp4"
FR = "/kaggle/tmp/frames"
os.makedirs(FR, exist_ok=True)

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

sh(f"{sys.executable} -m pip install -q gdown")
sh(f"gdown --id {FILE_ID} -O {SRC}")

report = {}
def stitch(fps, scale_w):
    d = f"{FR}/{fps}_{scale_w}"
    os.makedirs(d, exist_ok=True)
    sh(f"ffmpeg -hide_banner -loglevel error -i {SRC} -vf fps={fps},scale={scale_w}:-2 -q:v 2 {d}/f_%03d.jpg")
    files = sorted(glob.glob(f"{d}/*.jpg"))
    imgs = [cv2.imread(f) for f in files]
    st = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    st.setPanoConfidenceThresh(0.6)
    t = time.time()
    status, pano = st.stitch(imgs)
    key = f"fps{fps}_w{scale_w}"
    report[key] = {"frames": len(imgs), "status": int(status), "sec": round(time.time() - t, 1)}
    print(key, report[key], flush=True)
    if status == cv2.Stitcher_OK:
        report[key]["size"] = list(pano.shape[:2])
        cv2.imwrite(f"{OUT}/pano_{key}.jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return True
    return False

ok = False
for fps, w in [(2, 720), (1, 720), (3, 540), (1, 1080)]:
    try:
        if stitch(fps, w):
            ok = True
            if w == 720:
                break
    except Exception as e:
        report[f"fps{fps}_w{w}"] = {"error": str(e)}
        print("error", e, flush=True)

json.dump(report, open(f"{OUT}/stitch_report.json", "w"), indent=1)
print("OK" if ok else "FAILED", report)
