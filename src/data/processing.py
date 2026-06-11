import os
import sys
import glob
import shutil
import pandas as pd
from pathlib import Path
import re

# ---- Allow running from project root ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import (
    DATA_DIR, VIDEO_DIR, VIDEO_CSV_DIR, COMMENT_RAW_DIR,
    VIDEO_SEARCH, COMMENT_ALL, SPLIT_DATA_DIR, SPLIT_COMMENTS_DIR,
)


#合并各位博主视频csv
def merge_csv_files(input_folder, output_filepath):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    csv_pattern = os.path.join(input_folder, "*.csv")
    file_list = glob.glob(csv_pattern)

    if not file_list:
        print(f" '{input_folder}' 未找到CSV文件。")
        return

    file_list.sort()
    print(f"发现 {len(file_list)} 个文件，开始合并")

    df_list = []
    for file in file_list:
        if os.path.abspath(file) == os.path.abspath(output_filepath):
            continue

        try:
            df = pd.read_csv(file, encoding='utf-8')
            df_list.append(df)
            print(f"加载成功: {os.path.basename(file)} (行数: {len(df)})")
        except Exception as e:
            print(f"加载失败 {os.path.basename(file)}: {str(e)}")

    if df_list:
        merged_df = pd.concat(df_list, axis=0, ignore_index=True)
        try:
            merged_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
            print(f"\n合并完成。数据总规模: {len(merged_df)} 行，输出至: {output_filepath}")
        except Exception as e:
            print(f"\n写入异常: {str(e)}")
    else:
        print("\n无有效数据。")



#从视频标题与视频标签去筛选太极相关视频（merged_dy_data.csv）
def filter_taichi_videos(csv_file_path):
    backup_file_path = csv_file_path.replace('.csv', '_backup.csv')

    if not os.path.exists(csv_file_path):
        print("没找到需要筛选的原文件。")
        return

    try:
        shutil.copy2(csv_file_path, backup_file_path)
        print(f"原文件备份完成: {backup_file_path}")
    except Exception as e:
        print(f"备份出错了: {e}")
        return

    print("开始清洗非太极相关数据 (双重校验：标题+标签)...")
    try:
        df = pd.read_csv(csv_file_path, encoding='utf-8')

        # 校验两列是否都存在
        if '视频标签' not in df.columns or '视频标题' not in df.columns:
            print("表里缺少‘视频标签’或‘视频标题’列，请检查数据源格式。")
            return

        old_count = len(df)

        # 核心优化：使用 | (OR) 连接两个条件
        # 条件1：视频标签含有“太极”
        # 条件2：视频标题含有“太极”
        condition_tag = df['视频标签'].str.contains('太极', na=False)
        condition_title = df['视频标题'].str.contains('太极', na=False)

        # 只要满足其中一个条件，这一行就会被保留
        df = df[condition_tag | condition_title]

        new_count = len(df)

        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        print(f"双列筛选完毕。原本有 {old_count} 行，删了 {old_count - new_count} 行，保留了 {new_count} 行。")

    except Exception as e:
        print(f"筛选运行时发生错误: {e}")

#筛选太极相关视频去重（merged_dy_search.csv）
def filter_and_deduplicate_search_data(search_csv_path, main_csv_path):
    print("开始对搜索数据进行标题筛选、内部去重及跨文件去重...")

    if not os.path.exists(search_csv_path):
        print(f"找不到搜索数据文件: {search_csv_path}")
        return

    try:
        # 1. 读取搜索数据
        search_df = pd.read_csv(search_csv_path, encoding='utf-8-sig')

        if '视频标题' not in search_df.columns or '视频链接' not in search_df.columns:
            print("搜索表中缺少‘视频标题’或‘视频链接’这一列，无法执行过滤。")
            return

        original_count = len(search_df)

        # 2. 标题筛选：只保留视频标题中含有“太极”的行
        search_df = search_df[search_df['视频标题'].str.contains('太极', na=False)]
        after_title_filter_count = len(search_df)

        # 3. 内部链接去重：去除搜索表自带的重复项
        search_df = search_df.drop_duplicates(subset=['视频链接'], keep='first')
        after_internal_dedup_count = len(search_df)

        # 4. 跨文件去重：加载主表对比，剔除主库中已存在的链接
        cross_dedup_deleted = 0
        if os.path.exists(main_csv_path):
            main_df = pd.read_csv(main_csv_path, encoding='utf-8-sig')
            if '视频链接' in main_df.columns:
                existing_links = set(main_df['视频链接'].dropna().astype(str).str.strip())
                # 剔除存在于主表集合中的行
                search_df = search_df[~search_df['视频链接'].astype(str).str.strip().isin(existing_links)]
                cross_dedup_deleted = after_internal_dedup_count - len(search_df)
            else:
                print("⚠主表中没找到‘视频链接’列，跳过跨文件去重。")
        else:
            print(f"⚠找不到主表文件: {main_csv_path}，跳过跨文件去重。")

        final_count = len(search_df)

        # 5. 覆盖保存回原文件
        search_df.to_csv(search_csv_path, index=False, encoding='utf-8-sig')

        print(f"\n搜索数据清理汇总:")
        print(f"原始数据: {original_count} 行")
        print(f"因标题不含太极删去: {original_count - after_title_filter_count} 行")
        print(f"因内部链接重复删去: {after_title_filter_count - after_internal_dedup_count} 行")
        print(f"因主库已存在删去: {cross_dedup_deleted} 行")
        print(f"最终保留新有效数据: {final_count} 行")
        print("搜索数据深度过滤与去重完成！")

    except Exception as e:
        print(f"处理搜索数据时发生未知错误: {str(e)}")


#清除时长为0的视频记录及物理文件
def cleanup_zero_duration_content(csv_path, video_root_dir):
    print("开始检测并清理时长为0的视频及文件...")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df['视频时长'] = pd.to_numeric(df['视频时长'], errors='coerce')
        zero_indices = df[df['视频时长'] == 0].index

        if len(zero_indices) == 0:
            print("未发现视频时长为0的记录。")
            return

        deleted_files_total = 0

        for idx in zero_indices:
            author_name = str(df.loc[idx, '作者昵称'])
            video_link = str(df.loc[idx, '视频链接'])

            if video_link == 'nan' or not video_link.strip():
                continue

            video_id = video_link.split('/')[-1]
            author_dir = Path(video_root_dir) / author_name

            if author_dir.exists() and author_dir.is_dir():
                for video_file in author_dir.iterdir():
                    if video_id in video_file.name:
                        try:
                            video_file.unlink()
                            print(f"物理文件删除成功: {video_file.name}")
                            deleted_files_total += 1
                        except Exception as e:
                            print(f"文件 {video_file.name} 删除失败: {str(e)}")
            else:
                print(f"未找到作者文件夹: {author_name}")

        df_cleaned = df.drop(zero_indices)
        df_cleaned.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"\n时长0清理汇总:")
        print(f"1. 物理视频删除数量: {deleted_files_total}")
        print(f"2. CSV记录删除数量: {len(zero_indices)}")

    except FileNotFoundError:
        print(f"错误: 找不到文件路径 {csv_path}")
    except Exception as e:
        print(f"清理执行异常: {str(e)}")


#清除搜索数据中评论数小于20的行
def remove_low_comment_search_results(csv_path):
    print("开始清理评论数小于20的搜索视频记录...")
    try:
        if not os.path.exists(csv_path):
            print(f"没找到搜索数据文件: {csv_path}")
            return

        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        if '评论数' not in df.columns:
            print("表里没看到‘评论数’这一列，请检查数据格式。")
            return

        # 将评论数转换为数值类型，无法转换的变为NaN
        df['评论数'] = pd.to_numeric(df['评论数'], errors='coerce')

        # 找到评论数小于20的行索引（同时也把NaN视为无效数据一并处理）
        low_comment_indices = df[(df['评论数'] < 20) | (df['评论数'].isna())].index

        old_count = len(df)
        # 直接删除这些行
        df_cleaned = df.drop(low_comment_indices)
        new_count = len(df_cleaned)

        df_cleaned.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"评论数清理完毕。原搜索数据有 {old_count} 行，删除了 {old_count - new_count} 行，保留了 {new_count} 行。")

    except Exception as e:
        print(f"清理评论数时发生错误: {str(e)}")

#合并抖音博主视频爬取到的评论
def merge_comment_csv_files(input_folder, output_filepath):
    print("开始合并评论数据...")

    # 确保输出目录是存在的
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    csv_pattern = os.path.join(input_folder, "*.csv")
    file_list = glob.glob(csv_pattern)

    if not file_list:
        print(f"没在 '{input_folder}' 找到评论CSV文件。")
        return

    print(f"发现 {len(file_list)} 个评论文件，准备合并...")

    df_list = []
    for file in file_list:
        # 防止把已经合并好的输出文件又给读进去
        if os.path.abspath(file) == os.path.abspath(output_filepath):
            continue

        try:
            # 评论数据包含大量复杂文本和表情，统一用 utf-8 读取
            df = pd.read_csv(file, encoding='utf-8')
            df_list.append(df)
            print(f"读取成功: {os.path.basename(file)} (包含 {len(df)} 条评论)")
        except Exception as e:
            print(f"读取评论文件失败 {os.path.basename(file)}: {str(e)}")

    if df_list:
        # 按行直接拼在一起，重置行号
        merged_df = pd.concat(df_list, axis=0, ignore_index=True)

        try:
            # 写出时同样使用 utf-8-sig 防止 Excel 乱码
            merged_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
            print(f"评论合并完成。总计: {len(merged_df)} 条评论，文件保存在: {output_filepath}")
        except Exception as e:
            print(f"保存合并后的评论数据出错: {str(e)}")
    else:
        print("没有读到有效的评论数据。")

#清楚空评论的行以及全是@的行
def clean_invalid_comments(csv_path):
    print("开始清理空评论和纯@的无效评论...")

    if not os.path.exists(csv_path):
        print(f"找不到评论文件: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        if '评论内容' not in df.columns:
            print("表里没看到‘评论内容’这一列，检查一下数据格式。")
            return

        old_count = len(df)

        # 1. 直接剔除评论内容为 NaN 的行
        df = df.dropna(subset=['评论内容'])

        # 确保整列都是字符串类型，防止后续正则报错
        df['评论内容'] = df['评论内容'].astype(str)

        # 顺手剔除纯空格的行
        df = df[df['评论内容'].str.strip() != '']

        # 2. 剔除只包含 @xxx 的行
        def is_only_mentions(text):
            # 用正则把 @ 加上后面的非空白字符全部替换为空字符串
            cleaned_text = re.sub(r'@\S+', '', text)
            # 如果替换后什么都不剩了（或者只剩空格），说明这条评论全是 @
            return cleaned_text.strip() == ''

        # 取反（~），只保留那些“不全是@”的行
        df = df[~df['评论内容'].apply(is_only_mentions)]

        new_count = len(df)

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"评论清理完毕。原本有 {old_count} 行，删了 {old_count - new_count} 行，保留了 {new_count} 行。")

    except Exception as e:
        print(f"清理无效评论时发生错误: {str(e)}")


# 清理merged_dy_comments.csv中未在merged_dy_data.csv文件“视频链接”中出现的视频评论

def clean_orphaned_comments(video_csv_path, comment_csv_path):
    print("开始清理已被淘汰视频的孤立评论...")

    if not os.path.exists(video_csv_path):
        print(f"找不到视频数据基准文件: {video_csv_path}")
        return
    if not os.path.exists(comment_csv_path):
        print(f"找不到评论数据文件: {comment_csv_path}")
        return

    try:
        # 1. 读取视频主表，提取所有存活的“视频链接”
        video_df = pd.read_csv(video_csv_path, encoding='utf-8-sig')
        if '视频链接' not in video_df.columns:
            print("视频基准表中缺少‘视频链接’列，无法比对。")
            return

        # 将合法的视频链接提取出来并转为 set (集合)
        # 使用 set 的哈希表结构，能让后续的查找匹配速度提升百倍以上
        valid_video_links = set(video_df['视频链接'].dropna().astype(str))

        # 2. 读取评论表
        comment_df = pd.read_csv(comment_csv_path, encoding='utf-8-sig')
        if '目标链接' not in comment_df.columns:
            print("评论表中缺少‘目标链接’列，无法比对。")
            return

        old_count = len(comment_df)

        # 为了防止格式差异（如浮点型NaN和字符串的混淆），统一转为字符串并去除首尾空格
        comment_df['目标链接'] = comment_df['目标链接'].astype(str).str.strip()

        # 3. 核心比对与过滤：只保留“目标链接”存在于 valid_video_links 集合中的行
        comment_df = comment_df[comment_df['目标链接'].isin(valid_video_links)]

        new_count = len(comment_df)
        deleted_count = old_count - new_count

        # 4. 覆盖保存回原评论文件
        comment_df.to_csv(comment_csv_path, index=False, encoding='utf-8-sig')

        print(
            f"孤立评论清理完毕。原评论有 {old_count} 条，因为对应视频被删除了，跟着清理掉 {deleted_count} 条，最终保留 {new_count} 条有效评论。")

    except Exception as e:
        print(f"清理孤立评论时发生错误: {str(e)}")



# 清洗dy_file/dy_video中未在merged_dy_data.csv文件“视频链接或目标链接”中出现的视频

def clean_orphaned_local_videos(csv_path, video_root_dir):
    print("开始扫描并清理本地多余的物理视频文件...")

    if not os.path.exists(csv_path):
        print(f"找不到视频基准文件: {csv_path}")
        return
    if not os.path.exists(video_root_dir):
        print(f"找不到本地视频文件夹: {video_root_dir}")
        return

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if '目标链接' not in df.columns:
            print("视频基准表中缺少‘视频链接’列，无法提取视频ID。")
            return

        # 1. 提取所有合法的视频ID，存入集合(set)以提升后续检索速度
        valid_video_ids = set()
        for link in df['目标链接'].dropna():
            link_str = str(link).strip()
            if not link_str:
                continue

            # 处理 URL 提取数字 ID
            # 以防万一有类似 ?a=1 的参数，先用 ? 截断；再去掉末尾多余斜杠，最后取按 / 分隔的最后一段
            vid = link_str.split('?')[0].rstrip('/').split('/')[-1]
            if vid:
                valid_video_ids.add(vid)

        print(f"从 CSV 中成功提取了 {len(valid_video_ids)} 个唯一的有效视频 ID。")

        deleted_count = 0

        # 2. 遍历 dy_video 下的所有作者文件夹及文件
        for root, dirs, files in os.walk(video_root_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)

                # 检查当前文件名是否包含了我们集合里的任何一个有效视频ID
                # any() 函数一旦匹配到就会立刻返回 True，效率较高
                is_valid_file = any(vid in file_name for vid in valid_video_ids)

                # 如果文件名中找不到任何合法的 ID，执行删除
                if not is_valid_file:
                    try:
                        os.remove(file_path)
                        # 为了避免打印太多刷屏，这里只打印被删除的文件名
                        print(f"删除冗余物理文件: {file_name}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除文件 {file_name} 时遭遇系统拦截或报错: {e}")

        print(f"\n本地物理视频清理完毕。共删除了 {deleted_count} 个不再需要的视频文件。")

    except Exception as e:
        print(f"清理本地物理视频时发生错误: {str(e)}")


# 打平dy_video文件夹结构：将所有子文件夹中的视频移动到dy_video根目录
def flatten_video_directory(video_root_dir):
    print("开始打平视频文件夹结构，将所有视频移到根目录...\n")

    if not os.path.exists(video_root_dir):
        print(f"找不到视频根目录: {video_root_dir}")
        return

    video_root = Path(video_root_dir)
    moved_count = 0
    skipped_count = 0

    # 收集所有子目录中的文件（排除根目录本身的文件）
    for root, dirs, files in os.walk(video_root_dir):
        # 跳过根目录本身，只处理子文件夹
        if os.path.abspath(root) == os.path.abspath(video_root_dir):
            continue

        for file_name in files:
            src_path = os.path.join(root, file_name)
            dest_path = os.path.join(video_root_dir, file_name)

            # 处理重名：如果目标已存在，在文件名前加上原父文件夹名
            if os.path.exists(dest_path):
                parent_folder = os.path.basename(root)
                name, ext = os.path.splitext(file_name)
                new_name = f"{parent_folder}_{name}{ext}"
                dest_path = os.path.join(video_root_dir, new_name)
                # 万一加父文件夹名后还是重名，再加数字后缀
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(video_root_dir, f"{parent_folder}_{name}_{counter}{ext}")
                    counter += 1

            try:
                shutil.move(src_path, dest_path)
                print(f"已移动: {os.path.basename(root)}/{file_name} -> {os.path.basename(dest_path)}")
                moved_count += 1
            except Exception as e:
                print(f"移动失败 {file_name}: {e}")
                skipped_count += 1

    # 删除所有空的子文件夹
    for root, dirs, files in os.walk(video_root_dir, topdown=False):
        if os.path.abspath(root) == os.path.abspath(video_root_dir):
            continue
        try:
            # 仅当文件夹为空时删除
            if not os.listdir(root):
                os.rmdir(root)
                print(f"已删除空文件夹: {os.path.basename(root)}")
        except Exception as e:
            print(f"删除文件夹失败 {os.path.basename(root)}: {e}")

    print(f"\n打平完成！共移动 {moved_count} 个文件，跳过 {skipped_count} 个（可能因重名或其他错误）。")


# 按每1000行分割CSV文件，列保持不变
def split_csv_by_rows(csv_path, output_dir, rows_per_file=1000):
    """
    将CSV文件按指定行数分割为多个独立文件，列结构保持不变。
    :param csv_path: 源CSV文件路径
    :param output_dir: 分割后文件的输出目录
    :param rows_per_file: 每个文件包含的最大行数，默认1000
    """
    print(f"开始分割文件: {csv_path}")
    print(f"每个文件最多 {rows_per_file} 行，输出目录: {output_dir}")

    if not os.path.exists(csv_path):
        print(f"找不到源文件: {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        total_rows = len(df)
        total_files = (total_rows + rows_per_file - 1) // rows_per_file

        base_name = os.path.splitext(os.path.basename(csv_path))[0]

        print(f"总行数: {total_rows}，将生成 {total_files} 个文件")

        for i in range(total_files):
            start_idx = i * rows_per_file
            end_idx = min((i + 1) * rows_per_file, total_rows)
            chunk = df.iloc[start_idx:end_idx]

            output_filename = f"{base_name}_part{i + 1:04d}.csv"
            output_path = os.path.join(output_dir, output_filename)
            chunk.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"分割完成。共生成 {total_files} 个文件，保存在: {output_dir}")

    except Exception as e:
        print(f"分割CSV文件时发生错误: {str(e)}")


# 对split_all_comments下的所有CSV文件，只保留"评论内容"列
def keep_comment_column_only(input_dir):
    """
    遍历指定目录下的所有CSV文件，只保留"评论内容"列，删除其他所有列。
    处理后的文件直接覆盖原文件。
    :param input_dir: 包含CSV文件的目录路径
    """
    print(f"开始处理目录: {input_dir}")
    print("只保留'评论内容'列，删除其他所有列...")

    if not os.path.exists(input_dir):
        print(f"目录不存在: {input_dir}")
        return

    csv_pattern = os.path.join(input_dir, "*.csv")
    file_list = glob.glob(csv_pattern)

    if not file_list:
        print(f"在 '{input_dir}' 中未找到CSV文件。")
        return

    file_list.sort()
    total_files = len(file_list)
    print(f"发现 {total_files} 个CSV文件")

    for i, file in enumerate(file_list, 1):
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')

            if '评论内容' not in df.columns:
                print(f"[{i}/{total_files}] 跳过（无'评论内容'列）: {os.path.basename(file)}")
                continue

            original_cols = len(df.columns)
            df = df[['评论内容']]

            df.to_csv(file, index=False, encoding='utf-8-sig')
            print(f"[{i}/{total_files}] 已处理: {os.path.basename(file)} "
                  f"（{original_cols}列 -> 1列，{len(df)}行）")

        except Exception as e:
            print(f"[{i}/{total_files}] 处理失败 {os.path.basename(file)}: {str(e)}")

    print(f"\n处理完成！共处理 {total_files} 个文件。")


# 删除评论级别为"二级评论"的行
def remove_second_level_comments(csv_path):
    """
    读取CSV文件，删除"评论级别"列中内容为"二级评论"的所有行，并覆盖保存。
    :param csv_path: CSV文件路径
    """
    print("开始删除评论级别为'二级评论'的行...")

    if not os.path.exists(csv_path):
        print(f"找不到文件: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        if '评论级别' not in df.columns:
            print("表里缺少'评论级别'列，请检查数据源格式。")
            return

        old_count = len(df)

        # 删除"评论级别"为"二级评论"的行
        df = df[df['评论级别'] != '二级评论']

        new_count = len(df)
        deleted_count = old_count - new_count

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"二级评论删除完毕。原本有 {old_count} 行，删除了 {deleted_count} 行，保留了 {new_count} 行。")

    except Exception as e:
        print(f"删除二级评论时发生错误: {str(e)}")


# 删除merged_dy_data_Allcomments.csv中指定的列
def drop_comment_columns(csv_path):
    """
    删除评论CSV文件中指定的列：页码、评论者昵称、评论者id、评论者uid、评论者主页链接、评论级别。
    处理后的数据覆盖保存原文件。
    :param csv_path: CSV文件路径
    """
    columns_to_drop = ['页码', '评论者昵称', '评论者id', '评论者uid', '评论者主页链接', '评论级别']

    print("开始删除指定列...")
    print(f"要删除的列: {columns_to_drop}")

    if not os.path.exists(csv_path):
        print(f"找不到文件: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        original_cols = list(df.columns)
        print(f"原始列 ({len(original_cols)}): {original_cols}")

        # 只删除实际存在的列，忽略不存在的
        cols_found = [col for col in columns_to_drop if col in df.columns]
        cols_missing = [col for col in columns_to_drop if col not in df.columns]

        if cols_missing:
            print(f"以下列在原文件中不存在，已自动忽略: {cols_missing}")

        if not cols_found:
            print("没有需要删除的列，无需操作。")
            return

        df.drop(columns=cols_found, inplace=True)

        remaining_cols = list(df.columns)
        print(f"剩余列 ({len(remaining_cols)}): {remaining_cols}")

        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"列删除完成。已保存至: {csv_path}")

    except Exception as e:
        print(f"删除列时发生错误: {str(e)}")


if __name__ == '__main__':
    # Paths from centralized config
    RAW_CSV_FOLDER = str(VIDEO_CSV_DIR)
    MERGED_CSV_PATH = str(VIDEO_CSV_DIR / "merged_dy_data_Allcomments_labeled_cleaned_onehot.csv")
    VIDEO_BASE_DIR = str(VIDEO_DIR)
    SEARCH_CSV_PATH = str(VIDEO_SEARCH)

    COMMENT_FOLDER = str(COMMENT_RAW_DIR)
    MERGED_COMMENT_PATH = str(VIDEO_CSV_DIR / "merged_dy_data3_comments.csv")

    ALLCOMMENTS_CSV_PATH = str(COMMENT_ALL)

    SPLIT_ALLDATA_OUTPUT = str(SPLIT_DATA_DIR)
    SPLIT_ALLCOMMENTS_OUTPUT = str(SPLIT_COMMENTS_DIR)

    # # 合并操作
    # merge_csv_files(RAW_CSV_FOLDER, MERGED_CSV_PATH)
    #
    # filter_taichi_videos(MERGED_CSV_PATH)
    #
    # cleanup_zero_duration_content(MERGED_CSV_PATH, VIDEO_BASE_DIR)
    #
    # remove_low_comment_search_results(SEARCH_CSV_PATH)
    #
    # merge_comment_csv_files(COMMENT_FOLDER, MERGED_COMMENT_PATH)
    #
    # clean_invalid_comments(MERGED_COMMENT_PATH)
    #
    # clean_orphaned_comments(MERGED_CSV_PATH, MERGED_COMMENT_PATH)
    #
    clean_orphaned_local_videos(MERGED_CSV_PATH, VIDEO_BASE_DIR)
    #
    # filter_and_deduplicate_search_data(SEARCH_CSV_PATH,MERGED_CSV_PATH)
    #
    # flatten_video_directory(VIDEO_BASE_DIR)

    # # 按每1000行分割CSV文件
    # split_csv_by_rows(MERGED_CSV_PATH, SPLIT_ALLDATA_OUTPUT, rows_per_file=1000)
    # split_csv_by_rows(ALLCOMMENTS_CSV_PATH, SPLIT_ALLCOMMENTS_OUTPUT, rows_per_file=1000)

    # keep_comment_column_only(SPLIT_ALLCOMMENTS_OUTPUT)
    # drop_comment_columns(ALLCOMMENTS_CSV_PATH)
