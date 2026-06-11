"""
抖音豆包大模型视频标签识别
读取 unlabeled.csv，找到对应视频，通过 Files API 上传后用 doubao-seed 模型打标签
"""
import json
import os
import time
import csv
import sys
import re
import urllib3
from pathlib import Path

import requests

urllib3.disable_warnings()

# ---- Allow running from project root ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import VIDEO_DIR, VIDEO_CSV_SUB_DIR

# ---- 复用 xiaomi_label.py 的标签体系和提示词 ----
from src.labeling.xiaomi_label import (
    ONEHOT_LABELS,
    MULTILABEL_LABELS,
    build_columns,
    build_prompt,
    flatten_row,
    log_error,
    ERROR_LOG,
)

# ============================================================
# 配置
# ============================================================
UNLABELED_CSV = VIDEO_CSV_SUB_DIR / "unlabeled.csv"
CSV_OUTPUT_DIR = VIDEO_CSV_SUB_DIR
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("DOUYIN_API_KEY", "")
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-2-0-lite-260428"
FPS = 5

_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
}


def upload_video(file_path: Path) -> str | None:
    """上传视频文件到 Files API，返回 file_id"""
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/files",
                headers={"Authorization": f"Bearer {API_KEY}"},
                files={"file": (file_path.name, f, "video/mp4")},
                data={"purpose": "user_data", "preprocess_configs": json.dumps({"video": {"fps": FPS}})},
                timeout=120,
            )
        if resp.status_code != 200:
            print(f"  Upload HTTP {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        file_id = data["id"]

        # 轮询等待处理完成
        for _ in range(60):
            status_resp = requests.get(
                f"{BASE_URL}/files/{file_id}",
                headers=_HEADERS,
                timeout=30,
            )
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                status = status_data.get("status", "")
                if status in ("processed", "active"):
                    return file_id
                if status == "failed":
                    print(f"  File processing failed: {status_data}")
                    return None
            time.sleep(2)

        print(f"  File processing timeout")
        return None

    except Exception as e:
        print(f"  Upload error: {e}")
        return None


def call_api(file_id: str, video_id: str) -> dict | None:
    """调用豆包 responses API 进行标签识别"""
    body = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_video", "file_id": file_id},
                    {"type": "input_text", "text": build_prompt()},
                ],
            }
        ],
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                f"{BASE_URL}/responses",
                json=body,
                headers={**_HEADERS, "Content-Type": "application/json"},
                timeout=300,
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("output", []):
                    if item.get("role") == "assistant":
                        for block in item.get("content", []):
                            text = block.get("text", "")
                            if text:
                                return _parse_json_response(text, video_id)
                print(f"  Empty response, retrying...")
                if attempt < 2:
                    time.sleep(5)
            elif resp.status_code == 429:
                wait = (attempt + 1) * 30
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < 2:
                    time.sleep(10)
        except requests.Timeout:
            print(f"  Timeout (attempt {attempt + 1}/3)")
        except Exception as e:
            print(f"  API error: {e}")
            if attempt < 2:
                time.sleep(10)

    log_error(video_id, "failed after 3 retries")
    return None


def _parse_json_response(text: str, video_id: str) -> dict | None:
    """从模型返回文本中提取 JSON"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        print(f"  [!] JSON parse failed for {video_id}: {text[:200]}")
        log_error(video_id, f"JSON parse failed: {text[:200]}")
        return None


def main():
    global ERROR_LOG

    # ---- 解析命令行参数 ----
    batch_limit = 0  # 0 = 全部
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # ---- 读取 unlabeled.csv，提取 video_id ----
    if not UNLABELED_CSV.exists():
        print(f"文件不存在: {UNLABELED_CSV}")
        return

    ids: list[str] = []
    for line in UNLABELED_CSV.read_text(encoding="utf-8-sig").strip().split("\n")[1:]:
        url = line.split(",")[0].strip()
        if "/video/" in url:
            vid = url.rstrip("/").rsplit("/video/", 1)[-1]
            if vid.isdigit():
                ids.append(vid)

    ids = sorted(set(ids), key=int)
    print(f"从 unlabeled.csv 提取 {len(ids)} 个唯一视频 ID")

    # ---- 找到对应视频文件 ----
    mp4_map: dict[str, Path] = {}
    for mp4 in VIDEO_DIR.glob("*.mp4"):
        mp4_map[mp4.stem] = mp4

    matched = [vid for vid in ids if vid in mp4_map]
    missing = len(ids) - len(matched)
    print(f"匹配到 {len(matched)} 个视频文件" + (f"，{missing} 个缺失" if missing else ""))

    if not matched:
        print("没有可处理的视频")
        return

    # ---- 设置输出路径 ----
    first_id = matched[0]
    last_id = matched[-1]
    csv_path = CSV_OUTPUT_DIR / f"_douyin_{UNLABELED_CSV.stem}_{first_id}-{last_id}.csv"
    ERROR_LOG = CSV_OUTPUT_DIR / f"_douyin_{UNLABELED_CSV.stem}_{first_id}-{last_id}_errors.log"

    # ---- 断点续传 ----
    processed: set[str] = set()
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                processed = {row["video_id"].lstrip("\t") for row in reader}
        except Exception:
            pass

    pending = [vid for vid in matched if vid not in processed]
    if processed:
        print(f"已处理 {len(processed)}，剩余 {len(pending)}")

    if batch_limit > 0 and len(pending) > batch_limit:
        print(f"本批次限制 {batch_limit} 个（剩余 {len(pending)} 个）")
        pending = pending[:batch_limit]

    if not pending:
        print("没有需要处理的文件")
        return

    # ---- 初始化 CSV ----
    columns = build_columns()
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    if write_header:
        writer.writeheader()

    # ---- 逐文件处理 ----
    success = 0
    fail = 0
    try:
        for i, vid in enumerate(pending):
            mp4_path = mp4_map[vid]
            size_mb = mp4_path.stat().st_size / 1024 / 1024
            print(f"\n[{i + 1}/{len(pending)}] {vid} ({size_mb:.1f}MB)")

            # 上传视频
            file_id = upload_video(mp4_path)
            if file_id is None:
                fail += 1
                continue

            # 调用模型
            result = call_api(file_id, vid)
            if result is None:
                fail += 1
                continue

            # 写入 CSV
            try:
                row = flatten_row(vid, result)
                writer.writerow(row)
                csv_file.flush()
                success += 1
                print(f"  OK ({success} ok, {fail} fail)")
            except Exception as e:
                print(f"  Row error: {e}")
                fail += 1

    finally:
        csv_file.close()

    print(f"\n===== 完成 =====")
    print(f"成功: {success}, 失败: {fail}")
    print(f"CSV: {csv_path}")
    if ERROR_LOG and ERROR_LOG.exists():
        print(f"错误日志: {ERROR_LOG}")


if __name__ == "__main__":
    main()
