import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import VIDEO_DIR, VIDEO_BASE64_DIR

import base64
import concurrent.futures

VIDEO_DIR = VIDEO_DIR
OUTPUT_DIR = VIDEO_BASE64_DIR


def encode_one(video_path: Path) -> tuple[str, int]:
    """读取视频并编码为 base64，返回 (文件名, 字符数)"""
    raw = video_path.read_bytes()
    b64 = base64.b64encode(raw)
    text = b64.decode("utf-8")

    out_path = OUTPUT_DIR / f"{video_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    return video_path.name, len(text)


def main():
    if not VIDEO_DIR.exists():
        print(f"目录不存在: {VIDEO_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not mp4_files:
        print("未找到 .mp4 文件")
        return

    print(f"找到 {len(mp4_files)} 个视频文件\n")

    with concurrent.futures.ProcessPoolExecutor() as pool:
        futures = {pool.submit(encode_one, f): f for f in mp4_files}
        for future in concurrent.futures.as_completed(futures):
            name, chars = future.result()
            f = futures[future]
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"[完成] {name}  {size_mb:.1f}MB -> {chars:,} 字符")

    print(f"\n处理完毕: 输出到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
