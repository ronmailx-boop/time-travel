# Step 0a: download the source video, probe it, detect scenes and export small preview frames.
# Runs on Kaggle (internet on, no GPU needed). Outputs go to /kaggle/working.
import json, os, subprocess, sys

FILE_ID = os.environ.get("TT_FILE_ID", "18ayaR52j9BU0OVed-_8JfyIyi8ucq8BR")
OUT = "/kaggle/working"
SRC = "/kaggle/tmp/src.mp4"
os.makedirs("/kaggle/tmp", exist_ok=True)

def run(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

run(f"{sys.executable} -m pip install -q gdown")
run(f"gdown --id {FILE_ID} -O {SRC}")

probe = json.loads(run(f"ffprobe -v quiet -print_format json -show_format -show_streams {SRC}").stdout)
for s in probe.get("streams", []):
    s.pop("tags", None)
probe.get("format", {}).pop("tags", None)
json.dump(probe, open(f"{OUT}/probe.json", "w"), indent=1)

# Scene cuts
r = subprocess.run(
    f"ffmpeg -hide_banner -i {SRC} -vf \"select='gt(scene,0.3)',showinfo\" -an -f null -",
    shell=True, capture_output=True, text=True)
cuts = [float(l.split("pts_time:")[1].split()[0]) for l in r.stderr.splitlines() if "pts_time:" in l]
json.dump({"scene_cuts_sec": cuts}, open(f"{OUT}/scenes.json", "w"), indent=1)

# One frame every 2 seconds, 480px wide, as a contact sheet + individual frames
os.makedirs(f"{OUT}/frames", exist_ok=True)
run(f"ffmpeg -hide_banner -loglevel error -i {SRC} -vf fps=1/2,scale=480:-2 -q:v 4 {OUT}/frames/f_%03d.jpg")
run(f"ffmpeg -hide_banner -loglevel error -i {SRC} -vf fps=1/2,scale=320:-2,tile=6x6 -frames:v 1 -q:v 4 {OUT}/contact.jpg")
# Short low-res proxy clip (for building the player locally), no audio removal
run(f"ffmpeg -hide_banner -loglevel error -i {SRC} -vf scale=640:-2 -c:v libx264 -crf 30 -preset veryfast -c:a aac -b:a 96k {OUT}/proxy.mp4")
print("done", cuts)
