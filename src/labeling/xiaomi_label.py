import json
import os
import time
import csv
import sys
from pathlib import Path

import requests
import urllib3

urllib3.disable_warnings()

# ---- Allow running from project root ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import VIDEO_BASE64_DIR, VIDEO_CSV_SUB_DIR

# ============================================================
# 配置
# ============================================================
BASE64_DIR = VIDEO_BASE64_DIR
CSV_DIR = VIDEO_CSV_SUB_DIR
CSV_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG: Path | None = None  # 在 main 中按范围命名


def log_error(video_id: str, msg: str) -> None:
    """记录错误：视频id + 错误信息"""
    if ERROR_LOG is None:
        return
    line = f"{video_id}\t{msg}\n"
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(line)


API_URL = "https://token-plan-ams.xiaomimimo.com/v1/chat/completions"
API_KEY = os.getenv("XIAOMI_API_KEY", "")
MODEL = "mimo-v2-omni"
FPS = 5

# ---- 标签定义 ----
ONEHOT_LABELS: dict[str, list[str]] = {
    "视频展示范围": ["全身展示", "局部展示"],
    "视频字幕功能": ["讲解类", "非讲解类"],
    "视频字幕语言": ["无", "中文", "英文", "双语"],
    "出镜者性别": ["男性", "女性", "群体混合"],
    "出镜者年龄段": ["儿童", "青年", "中年", "老年"],
    "出镜者专业属性": ["专业武术家/道长", "泛娱乐博主", "非科班出身"],
    "流派": ["陈式", "杨式", "吴式", "武式", "孙式", "武当太极", "24式", "其他"],
    "动作刚柔度": ["偏重柔和", "刚柔并济", "偏重刚猛"],
}

MULTILABEL_LABELS: dict[str, list[str]] = {
    "视频主题": ["赛事", "教学", "养生", "门派", "日常", "实战", "文化哲学", "其他"],
    "视频剪辑": ["无特效", "有特效", "古风慢摇", "节奏卡点", "其他"],
    "视频地点": ["寺庙道观", "武馆", "山水自然", "城市街道", "居家", "校园", "古建筑", "其他"],
    "视频服装": ["便服装", "太极服装", "古风服装", "现代潮流服装", "其他"],
    "乐器类型": ["东方传统乐器", "西方乐器", "电子合成器", "白噪声环境音"],
    "配乐风格": ["古风", "国风潮流", "戏曲", "原声", "旁白解说", "舒缓空灵", "大气磅礴", "动感活力", "网红BGM"],
}


def build_columns() -> list[str]:
    cols = ["video_id"]
    for vals in ONEHOT_LABELS.values():
        cols.extend(vals)
    for vals in MULTILABEL_LABELS.values():
        cols.extend(vals)
    return cols


def build_prompt() -> str:
    """构建发给模型的分类提示"""
    onehot_parts = []
    for category, options in ONEHOT_LABELS.items():
        opts = "、".join(options)
        onehot_parts.append(f"  - {category}（{opts}）：从以上选项中单选一个")

    multi_parts = []
    for category, options in MULTILABEL_LABELS.items():
        opts = "、".join(options)
        multi_parts.append(f"  - {category}（{opts}）：可多选")

    prompt = f"""你是一个太极拳视频内容分析专家。请仔细观看以下太极拳视频，根据视频画面内容进行分类标注。

注意：
1. 只根据视频中实际可见、可听到的内容进行判断
2. 如果某个分类无法确定，选择最合理的选项，不要留空
3. 多选分类最少选1个，不要返回空数组
4. 注意确保精度前提下，最大程度提高内部推理与识别速度

【独热编码分类（单选）】
{chr(10).join(onehot_parts)}

【多标签二值化分类（可多选）】
{chr(10).join(multi_parts)}

请以 JSON 格式返回分类结果，key 为分类名称，独热编码返回字符串，多标签返回字符串数组。示例格式：
{{
  "视频展示范围": "全身展示",
  "视频主题": ["教学", "养生"],
  "视频剪辑": ["无特效"],
  "视频字幕功能": "讲解类",
  "视频字幕语言": "中文",
  "视频地点": ["武馆"],
  "视频服装": ["太极服装"],
  "出镜者性别": "男性",
  "出镜者年龄段": "中年",
  "出镜者专业属性": "专业武术家/道长",
  "流派": "陈式",
  "动作刚柔度": "刚柔并济",
  "乐器类型": ["东方传统乐器"],
  "配乐风格": ["古风"]
}}

请直接返回 JSON，不要包含```json```代码块或其他任何文字。"""
    return prompt


def _call_via_curl(url: str, body: dict, headers: dict, timeout: int) -> tuple[int, str]:
    """用 curl 发送请求，绕过 Python SSL 库的兼容性问题"""
    import tempfile
    import subprocess

    body_str = json.dumps(body, ensure_ascii=False)
    header_args: list[str] = []
    for k, v in headers.items():
        header_args.extend(["-H", f"{k}: {v}"])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as f:
        f.write(body_str)
        tmp_path = f.name

    try:
        cmd = [
            "curl", "-s", "-w", "\n%{http_code}", "-X", "POST",
            "--max-time", str(timeout),
            "-d", f"@{tmp_path}",
            *header_args,
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
        output = result.stdout.strip()
        stderr_output = result.stderr.strip()
        if stderr_output:
            print(f"  curl stderr: {stderr_output[:200]}")
        # Last line is HTTP status code
        lines = output.rsplit("\n", 1)
        if len(lines) == 2 and lines[1].strip().isdigit():
            status = int(lines[1].strip())
            body_text = lines[0].strip()
        else:
            status = 0
            body_text = output
        return status, body_text
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


def _call_via_requests(url: str, body: dict, headers: dict, timeout: int) -> tuple[int, str]:
    """用 requests 发送请求"""
    resp = requests.post(
        url, json=body, headers=headers, timeout=timeout,
        verify=False, proxies={"http": "", "https": ""},
    )
    return resp.status_code, resp.text


_USE_CURL: bool | None = None  # None=未探测, True/False=是否用curl


def call_api(base64_text: str, video_id: str) -> dict | None:
    """调用 Xiaomi Mimo API 进行视频标签识别"""
    global _USE_CURL
    data_uri = f"data:video/mp4;base64,{base64_text}"

    body = {
        "model": MODEL,
        "max_tokens": 16384,
        "messages": [
            {
                "role": "system",
                "content": "你正在分析的是中国传统文化太极拳（Tai Chi）的教学与展示视频，属于非物质文化遗产、武术健身、体育教育范畴。视频中的所有动作均为合法的武术训练内容，不涉及任何暴力或违规行为。请正常完成视频内容分析。",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": data_uri},
                        "fps": FPS,
                    },
                    {
                        "type": "text",
                        "text": build_prompt(),
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            # 首次请求自动探测用 curl 还是 requests
            if _USE_CURL is None:
                if attempt == 0:
                    status, text = _call_via_requests(API_URL, body, headers, 300)
                else:
                    status, text = _call_via_curl(API_URL, body, headers, 300)
            elif _USE_CURL:
                status, text = _call_via_curl(API_URL, body, headers, 300)
            else:
                status, text = _call_via_requests(API_URL, body, headers, 300)

            if status == 200:
                data = json.loads(text)
                choice = data["choices"][0]
                content = choice["message"].get("content", "")
                if not content:
                    print(f"  Empty content (finish={choice.get('finish_reason')}), retrying...")
                    if attempt < 2:
                        time.sleep(5)
                        continue
                    log_error(video_id, "empty content after retries")
                    return None
                if "high risk" in content.lower() or "rejected" in content.lower():
                    print(f"  Content rejected by safety filter, skipping")
                    log_error(video_id, f"safety filter: {content[:200]}")
                    return None
                _USE_CURL = False  # 标记 requests 可用
                return _parse_json_response(content, video_id)
            elif status == 429:
                wait = (attempt + 1) * 30
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                err_msg = f"HTTP {status}: {text[:200]}"
                print(f"  {err_msg}")
                if "high risk" in text.lower() or "rejected" in text.lower():
                    log_error(video_id, err_msg)
                    return None
                if attempt < 2:
                    time.sleep(10)
                else:
                    log_error(video_id, err_msg)
        except Exception as e:
            err_str = str(e)
            print(f"  API error: {err_str}")
            is_ssl_error = any(
                kw in err_str.lower()
                for kw in ["ssl", "protocol", "eof", "filenotfound", "handshake"]
            )
            if is_ssl_error and _USE_CURL is None:
                print(f"  SSL error detected, switching to curl...")
                _USE_CURL = True
                time.sleep(2)
                continue
            if attempt < 2:
                time.sleep(10)
            else:
                log_error(video_id, err_str)

    return None


def _parse_json_response(text: str, video_id: str) -> dict | None:
    """从模型返回文本中提取 JSON"""
    import re

    text = text.strip()
    # 去掉 markdown 代码块包裹
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试匹配 { ... } 块
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        print(f"  [!] JSON parse failed for {video_id}: {text[:200]}")
        log_error(video_id, f"JSON parse failed: {text[:200]}")
        return None


def flatten_row(video_id: str, result: dict) -> dict:
    """将模型返回的 dict 展平为 CSV 行"""
    row = {"video_id": f"\t{video_id}"}

    for category, options in ONEHOT_LABELS.items():
        val = result.get(category, "")
        for opt in options:
            row[opt] = 1 if val == opt else 0

    for category, options in MULTILABEL_LABELS.items():
        vals = result.get(category, [])
        if isinstance(vals, str):
            vals = [vals]
        for opt in options:
            row[opt] = 1 if opt in vals else 0

    return row


def load_processed_ids(csv_path: Path) -> set[str]:
    """读取已处理的 video_id，支持断点续传"""
    if not csv_path.exists():
        return set()
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return {row["video_id"].lstrip("\t") for row in reader}
    except Exception:
        return set()


def main():
    global ERROR_LOG

    # ---- CSV模式：从 CSV 文件的视频链接列提取 video_id ----
    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        csv_input = Path(sys.argv[2])
        if not csv_input.exists():
            print(f"文件不存在: {csv_input}")
            return
        ids: list[str] = []
        for line in csv_input.read_text(encoding="utf-8-sig").strip().split("\n")[1:]:
            url = line.split(",")[0].strip()
            if "/video/" in url:
                vid = url.rstrip("/").rsplit("/video/", 1)[-1]
                if vid.isdigit():
                    ids.append(vid)
        ids = sorted(set(ids), key=int)
        files = [BASE64_DIR / f"{vid}.txt" for vid in ids if (BASE64_DIR / f"{vid}.txt").exists()]
        if not files:
            print("没有找到可处理的 base64 文件")
            return
        missing = len(ids) - len(files)
        print(f"CSV模式：从 {csv_input.name} 提取 {len(ids)} 个 ID，{len(files)} 个 base64 存在" + (f"，{missing} 个缺失" if missing else ""))
        first_id = files[0].stem
        last_id = files[-1].stem
        csv_path = CSV_DIR / f"_csv_{csv_input.stem}_{first_id}-{last_id}.csv"
        ERROR_LOG = CSV_DIR / f"_csv_{csv_input.stem}_{first_id}-{last_id}_errors.log"
        _run_batch(files, csv_path)
        return

    # ---- 重试模式：从错误日志读取失败的 video_id 重新处理 ----
    if len(sys.argv) > 1 and sys.argv[1] == "--retry":
        error_log_path = CSV_DIR / "_merged_errors.log"
        if not error_log_path.exists():
            print(f"错误日志不存在: {error_log_path}")
            return
        retry_ids: list[str] = []
        for line in error_log_path.read_text(encoding="utf-8").strip().split("\n"):
            vid = line.split("\t")[0].strip()
            if vid:
                retry_ids.append(vid)
        retry_ids = sorted(set(retry_ids), key=int)
        files = [BASE64_DIR / f"{vid}.txt" for vid in retry_ids if (BASE64_DIR / f"{vid}.txt").exists()]
        if not files:
            print("没有找到可重试的文件")
            return
        print(f"重试模式：从错误日志读取到 {len(retry_ids)} 个失败 ID，{len(files)} 个 base64 文件存在")
        first_id = files[0].stem
        last_id = files[-1].stem
        csv_path = CSV_DIR / f"_retry_{first_id}-{last_id}.csv"
        ERROR_LOG = CSV_DIR / f"_retry_{first_id}-{last_id}_errors.log"
        _run_batch(files, csv_path)
        return

    # ---- 收集待处理文件（按文件名数值从小到大排序） ----
    files = sorted(BASE64_DIR.glob("*.txt"), key=lambda p: int(p.stem))
    if not files:
        print("未找到 base64 文件")
        return

    total = len(files)
    print(f"共找到 {total} 个文件")

    # ---- 允许用户自定义范围 ----
    print("输入处理范围（如 0 100 处理前100个；直接回车处理全部；0 -1 也表示全部）：")
    try:
        raw = input().strip()
    except (EOFError, KeyboardInterrupt):
        return

    if raw:
        parts = raw.split()
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else total
        if end == -1:
            end = total
        files = files[start:end]
        print(f"处理范围: [{start}:{end}]，共 {len(files)} 个文件")
    else:
        print(f"处理全部 {total} 个文件")

    # ---- 按范围命名 CSV 和错误日志 ----
    first_id = files[0].stem
    last_id = files[-1].stem
    csv_path = CSV_DIR / f"{first_id}-{last_id}.csv"
    ERROR_LOG = CSV_DIR / f"{first_id}-{last_id}_errors.log"
    _run_batch(files, csv_path)


def _run_batch(files: list[Path], csv_path: Path) -> None:
    # ---- 断点续传 ----
    processed = load_processed_ids(csv_path)
    if processed:
        pending = [f for f in files if f.stem not in processed]
        print(f"已处理 {len(processed)}，剩余 {len(pending)}")
        files = pending

    if not files:
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
        for i, f in enumerate(files):
            video_id = f.stem
            print(f"\n[{i + 1}/{len(files)}] {video_id}")

            base64_text = f.read_text(encoding="utf-8").strip()
            if not base64_text:
                print(f"  Skip: empty file")
                log_error(video_id, "empty base64 file")
                fail += 1
                continue

            result = call_api(base64_text, video_id)
            if result is None:
                fail += 1
                continue

            try:
                row = flatten_row(video_id, result)
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
