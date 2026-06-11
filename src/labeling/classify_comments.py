"""
对split_all_comments文件夹下的评论CSV文件进行LLM多标签分类。
使用DeepSeek API (OpenAI兼容接口) 一次性上传整个CSV文件，
由大模型统一判读后为每条评论添加四组标签列。

研究背景：数据收集于抖音平台，主题为太极拳相关视频评论，
用于后续论文论点研究的标签标注工作。

标签体系（列名带前缀以避免重名"其他"）：
  第一组（情感倾向 - 独热编码）：情感_肯定/赞美、情感_诋毁/谩骂、情感_客观中立
  第二组（内容属性1 - 多标签二值化）：属性一_尊师重道、属性一_实战性质疑、属性一_文化质疑、
      属性一_文化捍卫、属性一_玩网络梗/看热闹、属性一_捧杀阴阳怪气、
      属性一_对博主人身攻击、属性一_对博主性骚扰、属性一_其他
  第三组（内容属性2 - 多标签二值化）：属性二_关注动作要点、属性二_关注实战技术、属性二_关注养生功效、
      属性二_关注视觉审美、属性二_关注文化哲学、属性二_关注博主本身、属性二_关注其他人物、
      属性二_关注配乐审美、属性二_关注跨拳种实战比较、属性二_其他
  第四组（行为意向 - 多标签二值化）：意向_种草意向、意向_资源索取、意向_技术追问、
      意向_文化探讨、意向_跟练体感、意向_病理/生理求助、意向_寻师求学、
      意向_主动推荐、意向_影视文学跨界探讨、意向_门派流派探讨、意向_其他
"""

import os
import sys
import json
import time
import re
import pandas as pd
import requests
from pathlib import Path

# ---- Allow running from project root ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import SPLIT_COMMENTS_DIR, SPLIT_COMMENTS_LABELED_DIR

# ========== 配置 ==========
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-flash"

# 输入输出路径
INPUT_DIR = SPLIT_COMMENTS_DIR
OUTPUT_DIR = SPLIT_COMMENTS_LABELED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API配置
MAX_TOKENS = 16384          # 单次输出上限（一次处理1000条需足够的输出tokens）
MAX_RETRIES = 2             # 失败重试次数
REQUEST_TIMEOUT = 300       # 单次请求超时（秒），1000条评论处理需要更长时间

# ========== 标签定义 ==========
LABEL_SETS = {
    "sentiment": {
        "key": "情感倾向",
        "labels": ["肯定/赞美", "诋毁/谩骂", "客观中立"],
        "encoding": "one-hot",
        "prefix": "情感_",
    },
    "content_cat1": {
        "key": "内容属性一",
        "labels": ["尊师重道", "实战性质疑", "文化质疑", "文化捍卫",
                    "玩网络梗", "阴阳怪气", "对博主人身攻击",
                    "对博主性骚扰", "其他"],
        "encoding": "multi-label",
        "prefix": "属性一_",
    },
    "content_cat2": {
        "key": "内容属性二",
        "labels": ["动作要点", "实战技术", "养生功效", "视觉审美",
                    "文化哲学", "博主本身", "关注其他人物",
                    "关注配乐审美", "关注跨拳种实战比较", "其他"],
        "encoding": "multi-label",
        "prefix": "属性二_",
    },
    "intent": {
        "key": "行为意向",
        "labels": ["种草意向", "资源索取", "技术追问", "文化探讨",
                    "跟练体感", "病理/生理求助", "寻师求学", "主动推荐",
                    "影视文学跨界探讨", "门派流派探讨", "其他"],
        "encoding": "multi-label",
        "prefix": "意向_",
    },
}

# 紧凑格式输出时用到的标签短码
# 格式: index|sentiment|attr1,attr1,...|attr2,attr2,...|intent,intent,...
# 例如: 0|肯定/赞美|文化捍卫||技术追问
# 空值用空字符串，各标签组之间用|分隔，组内多标签用英文逗号分隔


def get_all_label_columns() -> list:
    """获取所有标签的完整列名（带前缀）"""
    cols = []
    for label_set in LABEL_SETS.values():
        prefix = label_set["prefix"]
        for label in label_set["labels"]:
            cols.append(f"{prefix}{label}")
    return cols


ALL_LABEL_COLUMNS = get_all_label_columns()

SYSTEM_PROMPT = """你是一个精确的中文社交媒体评论分类器。

## 研究背景
这些评论收集于抖音（Douyin）平台上关于太极拳相关视频的评论。我们正在进行学术研究，目的是为后续论文的论点分析添加结构化的标签数据。请从学术研究的角度，严谨地对每条评论进行分类标注。

## 标签体系
你需要为每条评论输出四组标签：

### 第一组：情感倾向（三选一，独热编码）
- 肯定/赞美：表达认可、赞美、支持、感谢等正面情绪
- 诋毁/谩骂：表达贬低、侮辱、嘲讽、攻击等负面情绪
- 客观中立：陈述事实、理性讨论、无明显情绪倾向

### 第二组：内容属性-类别一（多标签，可多选）
- 尊师重道：表达对武术传统、师承、门派、规矩的尊重
- 实战性质疑：质疑武术实战能力、规则合理性、比赛真实性
- 文化质疑：质疑武术文化价值、传统意义、现代适用性
- 文化捍卫：为武术/传统文化辩护、反驳质疑
- 玩网络梗：引用网络梗、娱乐消遣心态
- 阴阳怪气：明褒暗贬、阴阳怪气、反讽
- 对博主人身攻击：针对博主本人而非内容的攻击
- 对博主性骚扰：带有性暗示/骚扰性质的言论
- 其他：以上类别均不符合

### 第三组：内容属性-类别二（多标签，可多选）
- 动作要点：讨论具体动作要领、姿势、发力方式
- 实战技术：讨论实战技巧、攻防策略、对抗技术
- 养生功效：讨论健康养生效果、身体改善
- 视觉审美：讨论动作美感、观赏性、服装造型
- 文化哲学：讨论武术哲学、文化内涵、精神层面
- 博主本身：讨论博主个人特质（非攻击性）
- 关注其他人物：讨论视频中出现的其他人物（非博主本人）
- 关注配乐审美：讨论背景音乐、音效、配乐风格
- 关注跨拳种实战比较：对比太极拳与其他拳种/格斗技的实战能力
- 其他：以上类别均不符合

### 第四组：行为意向（多标签，可多选）
- 种草意向：表示被内容吸引、想尝试/模仿
- 资源索取：求教程、求链接、求资源
- 技术追问：追问技术细节、方法步骤
- 文化探讨：深入探讨文化相关问题
- 跟练体感：分享自己跟练/练习的体验感受
- 病理/生理求助：咨询健康问题、身体状况
- 寻师求学：表达拜师、求学、找教练意愿
- 主动推荐：向他人推荐该内容或博主
- 影视文学跨界探讨：讨论影视、文学作品中与太极拳相关的内容
- 门派流派探讨：讨论太极拳不同门派/流派的差异与特点
- 其他：以上类别均不符合

## 输出格式
逐行输出每条评论的分类结果，每行一条，格式为（用|分隔五个字段）：
索引|情感倾向|内容属性一(多标签用逗号分隔)|内容属性二(多标签用逗号分隔)|行为意向(多标签用逗号分隔)

示例：
0|客观中立|尊师重道,文化捍卫|动作要点|
1|肯定/赞美||实战技术|技术追问,种草意向
2|诋毁/谩骂|实战性质疑,文化质疑|实战技术|

## 注意事项
1. 只输出上述格式的行，不要额外文字
2. 必须为每条评论都输出一行
3. 索引必须与输入编号完全对应
4. 每组都必须做出选择，不可留空的情感倾向会默认为"客观中立"
5. 如某组确实无匹配标签，该位置留空（如示例中的"||"）
6. 多标签组内用英文逗号分隔，不要有空格"""


def build_prompt_with_all_comments(comments: list[str]) -> str:
    """构建包含所有评论的用户提示词"""
    lines = [f"以下是从抖音平台收集的 {len(comments)} 条太极拳相关视频评论，请逐一进行分类标注：\n"]
    for i, comment in enumerate(comments):
        lines.append(f"[{i}] {comment}")
    return "\n".join(lines)


def call_api_for_entire_file(comments: list[str]) -> list[dict]:
    """
    一次性将整个CSV文件的所有评论提交给DeepSeek API进行分类。
    返回解析后的结果列表。
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    user_prompt = build_prompt_with_all_comments(comments)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            print(f"  发送API请求... (评论数: {len(comments)}, 尝试: {attempt+1}/{MAX_RETRIES+1})")
            start = time.time()

            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json()

            elapsed = time.time() - start
            print(f"  API响应耗时: {elapsed:.1f}s")

            content = result["choices"][0]["message"]["content"].strip()

            # 解析逐行格式的结果
            parsed = parse_compact_format(content, len(comments))

            if parsed:
                # 检查完整性
                found_indices = set(r["index"] for r in parsed)
                missing = len(comments) - len(found_indices)
                if missing > len(comments) * 0.1:  # 缺失超过10%则重试
                    print(f"  警告: 仅解析出 {len(parsed)}/{len(comments)} 条结果（缺失 {missing} 条），将重试...")
                    if attempt < MAX_RETRIES:
                        time.sleep(3)
                        continue
                else:
                    print(f"  成功解析 {len(parsed)}/{len(comments)} 条结果")
                    return parsed
            else:
                print(f"  解析失败，未能获取有效结果，重试...")
                if attempt < MAX_RETRIES:
                    time.sleep(3)
                    continue

        except requests.exceptions.Timeout:
            print(f"  API请求超时 ({REQUEST_TIMEOUT}s)，重试... ({attempt+1}/{MAX_RETRIES+1})")
            if attempt < MAX_RETRIES:
                time.sleep(5)
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP错误: {e}")
            print(f"  响应内容: {result.get('text', '')[:500] if 'result' in dir() else 'N/A'}")
            if attempt < MAX_RETRIES:
                time.sleep(3)
        except Exception as e:
            print(f"  API调用失败 ({attempt+1}/{MAX_RETRIES+1}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(3)

    print(f"  全部重试失败，返回空结果")
    return []


def parse_compact_format(content: str, num_expected: int) -> list[dict]:
    """
    解析紧凑格式的输出：
    索引|情感倾向|属性一(逗号分隔)|属性二(逗号分隔)|意向(逗号分隔)
    返回 list of dict
    """
    results = []
    content = content.strip()

    # 移除可能的markdown代码块标记
    if content.startswith("```"):
        content = re.sub(r'^```\w*\n?', '', content)
        content = re.sub(r'\n?```$', '', content)
        content = content.strip()

    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        parts = line.split("|")
        if len(parts) < 3:  # 至少需要 index + 情感倾向 + (可为空的其余字段)
            continue

        try:
            idx = int(parts[0].strip())
            sentiment = parts[1].strip() if parts[1].strip() else "客观中立"
            attr1_raw = parts[2].strip() if len(parts) > 2 else ""
            attr2_raw = parts[3].strip() if len(parts) > 3 else ""
            intent_raw = parts[4].strip() if len(parts) > 4 else ""

            # 解析各多标签组
            attr1 = [x.strip() for x in attr1_raw.split(",") if x.strip()] if attr1_raw else []
            attr2 = [x.strip() for x in attr2_raw.split(",") if x.strip()] if attr2_raw else []
            intent = [x.strip() for x in intent_raw.split(",") if x.strip()] if intent_raw else []

            results.append({
                "index": idx,
                "情感倾向": sentiment,
                "内容属性一": attr1,
                "内容属性二": attr2,
                "行为意向": intent,
            })
        except ValueError:
            # 无法解析索引的行，跳过
            continue

    return results


def flatten_labels(parsed_results: list, num_expected: int) -> pd.DataFrame:
    """将API返回的分类结果展平为带前缀的标签DataFrame"""
    # 建立索引到结果的映射
    result_map = {r["index"]: r for r in parsed_results}

    rows = []
    for i in range(num_expected):
        row = {col: 0 for col in ALL_LABEL_COLUMNS}

        matched = result_map.get(i)
        if matched is None:
            rows.append(row)
            continue

        # 第一组：情感倾向（独热编码）
        sentiment = matched.get("情感倾向", "客观中立")
        prefix = LABEL_SETS["sentiment"]["prefix"]
        labels = LABEL_SETS["sentiment"]["labels"]
        if sentiment in labels:
            row[f"{prefix}{sentiment}"] = 1
        else:
            row[f"{prefix}客观中立"] = 1

        # 第二组：内容属性一（多标签）
        _apply_multilabel(row, matched.get("内容属性一", []),
                          LABEL_SETS["content_cat1"]["prefix"],
                          LABEL_SETS["content_cat1"]["labels"])

        # 第三组：内容属性二（多标签）
        _apply_multilabel(row, matched.get("内容属性二", []),
                          LABEL_SETS["content_cat2"]["prefix"],
                          LABEL_SETS["content_cat2"]["labels"])

        # 第四组：行为意向（多标签）
        _apply_multilabel(row, matched.get("行为意向", []),
                          LABEL_SETS["intent"]["prefix"],
                          LABEL_SETS["intent"]["labels"])

        rows.append(row)

    return pd.DataFrame(rows, columns=ALL_LABEL_COLUMNS)


def _apply_multilabel(row: dict, matched_labels: list, prefix: str, valid_labels: list):
    """将LLM返回的多标签列表填入row字典"""
    for label in matched_labels:
        label = label.strip()
        if label in valid_labels:
            row[f"{prefix}{label}"] = 1
        elif label:
            # 无法匹配的标签统一归入"其他"
            other_key = f"{prefix}其他"
            if other_key in row:
                row[other_key] = 1


def classify_csv_file(input_filepath: str, output_filepath: str):
    """
    对单个CSV文件进行评论分类。
    一次性将文件中所有评论提交给API，由大模型统一判读后返回全部分类结果。
    """
    print(f"\n{'='*60}")
    print(f"处理文件: {os.path.basename(input_filepath)}")
    print(f"策略: 一次性上传完整文件，由大模型统一判读")
    print(f"{'='*60}")

    df = pd.read_csv(input_filepath, encoding='utf-8')
    total_rows = len(df)
    print(f"总行数: {total_rows}")

    if "评论内容" not in df.columns:
        print("错误: 找不到'评论内容'列！")
        return

    comments = df["评论内容"].fillna("").astype(str).tolist()

    # 初始化标签列
    for col in ALL_LABEL_COLUMNS:
        df[col] = 0

    # 一次性API调用
    start_time = time.time()
    parsed = call_api_for_entire_file(comments)
    elapsed = time.time() - start_time

    if parsed:
        # 将结果填入df
        labels_df = flatten_labels(parsed, len(comments))
        for i in range(len(comments)):
            for col in ALL_LABEL_COLUMNS:
                df.at[i, col] = labels_df.iloc[i][col]

        # 统计覆盖率
        covered = len(set(r["index"] for r in parsed))
        print(f"\n  处理完成！覆盖率: {covered}/{total_rows} ({covered/total_rows*100:.1f}%), "
              f"总耗时: {elapsed:.1f}s")
    else:
        print(f"\n  处理失败！未获取到有效分类结果。耗时: {elapsed:.1f}s")

    # 保存结果
    df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {output_filepath}")

    # 统计标签分布
    print("\n标签分布统计:")
    for col in ALL_LABEL_COLUMNS:
        count = int(df[col].sum())
        if count > 0:
            pct = count / total_rows * 100
            print(f"  {col}: {count} ({pct:.1f}%)")

    return df


def main(count: int = None, skip: int = 0):
    """
    处理 split_all_comments 下的CSV文件。

    参数:
        count: 处理前N个文件（None=全部）
        skip: 跳过前M个文件，从第M+1个开始（与count配合使用）
    """
    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    # 过滤掉测试文件（以_开头的文件）
    csv_files = [f for f in csv_files if not f.name.startswith("_")]

    if not csv_files:
        print(f"错误: 在 {INPUT_DIR} 中未找到CSV文件")
        return

    total_available = len(csv_files)

    # 截取指定范围的切片
    if skip > 0:
        csv_files = csv_files[skip:]
    if count is not None:
        csv_files = csv_files[:count]

    if not csv_files:
        print(f"错误: 指定的范围 (skip={skip}, count={count}) 内没有文件")
        return

    print(f"总文件数: {total_available}, 本次处理: {len(csv_files)} 个")
    if skip > 0:
        print(f"  跳过前 {skip} 个，从第 {skip+1} 个开始")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"策略: 每个文件整体上传一次，由大模型统一判读后返回所有标签")
    print(f"{'#'*60}\n")

    total_start = time.time()
    success_count = 0
    fail_count = 0

    for i, input_file in enumerate(csv_files, 1):
        output_file = OUTPUT_DIR / input_file.name.replace(".csv", "_labeled.csv")

        print(f"\n{'#'*60}")
        print(f"[{i}/{len(csv_files)}]  (全局序号: {skip + i}/{total_available})")
        print(f"{'#'*60}")

        try:
            classify_csv_file(str(input_file), str(output_file))
            success_count += 1
        except Exception as e:
            print(f"  处理文件失败: {e}")
            fail_count += 1

        # 文件处理之间短暂休息，避免API限流
        if i < len(csv_files):
            time.sleep(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"本次处理完成！")
    print(f"成功: {success_count}, 失败: {fail_count}")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


def test_single_file():
    """测试模式：仅处理 test 文件"""
    test_file = INPUT_DIR / "_test_part0001.csv"
    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return
    output_file = OUTPUT_DIR / "_test_part0001_labeled.csv"
    classify_csv_file(str(test_file), str(output_file))
    print(f"\n测试完成！输出: {output_file}")


def print_usage():
    """打印使用说明"""
    print(f"""
用法: python classify_comments.py [选项]

选项:
  --test              测试模式，仅处理测试文件
  -n N, --count N     处理前 N 个文件（不指定则处理全部）
  -s N, --skip N      跳过前 N 个文件（默认 0）
  -h, --help          显示此帮助

示例:
  python classify_comments.py -n 5          # 处理前5个文件
  python classify_comments.py -n 10 -s 5    # 跳过前5个，处理第6-15个文件
  python classify_comments.py               # 处理全部文件
  python classify_comments.py --test        # 测试模式
""")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    count = None
    skip = 0
    i = 0

    while i < len(args):
        arg = args[i]
        if arg in ("-h", "--help"):
            print_usage()
            sys.exit(0)
        elif arg == "--test":
            test_single_file()
            sys.exit(0)
        elif arg in ("-n", "--count"):
            i += 1
            if i < len(args):
                try:
                    count = int(args[i])
                except ValueError:
                    print(f"错误: -n 参数必须是整数，收到: {args[i]}")
                    sys.exit(1)
            else:
                print("错误: -n 参数缺少值")
                sys.exit(1)
        elif arg in ("-s", "--skip"):
            i += 1
            if i < len(args):
                try:
                    skip = int(args[i])
                except ValueError:
                    print(f"错误: -s 参数必须是整数，收到: {args[i]}")
                    sys.exit(1)
            else:
                print("错误: -s 参数缺少值")
                sys.exit(1)
        else:
            print(f"未知参数: {arg}")
            print_usage()
            sys.exit(1)
        i += 1

    main(count=count, skip=skip)
