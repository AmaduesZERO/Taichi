import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import VIDEO_DIR, FFMPEG, FFPROBE

import subprocess
import json

VIDEO_DIR = VIDEO_DIR
LOG_FILE = VIDEO_DIR / "trim_errors.log"
FFMPEG = str(FFMPEG)
FFPROBE = str(FFPROBE)

errors: list[str] = []


def get_fps(input_path: Path) -> float | None:
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        result.check_returncode()
    except subprocess.CalledProcessError:
        msg = f"ffprobe error: {input_path.name}"
        print(f"  {msg}")
        errors.append(msg)
        return None
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        msg = f"no video stream: {input_path.name}"
        print(f"  {msg}")
        errors.append(msg)
        return None
    fps_str = streams[0]["r_frame_rate"]
    num, den = fps_str.split("/")
    return float(num) / float(den)


def convert_to_10fps(input_path: Path) -> None:
    tmp_path = input_path.with_suffix(".tmp" + input_path.suffix)
    cmd = [
        FFMPEG, "-i", str(input_path),
        "-filter:v", "fps=10",
        "-c:a", "copy",
        "-y",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_path.replace(input_path)
    print(f"Done: {input_path.name}")


def main() -> None:
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for p in VIDEO_DIR.iterdir():
        if p.suffix.lower() not in video_extensions:
            continue
        fps = get_fps(p)
        if fps is None:
            continue
        if fps <= 10:
            print(f"Skip (fps={fps:.1f}): {p.name}")
            continue
        print(f"Processing (fps={fps:.1f}): {p.name}")
        convert_to_10fps(p)

    if errors:
        LOG_FILE.write_text("\n".join(errors), encoding="utf-8")
        print(f"\n{len(errors)} errors logged to: {LOG_FILE}")


if __name__ == "__main__":
    main()
