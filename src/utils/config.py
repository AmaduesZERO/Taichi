"""
Centralised path configuration for the TaiChi project.
All scripts import paths from here — no more hardcoded absolute paths.

Usage:
    from src.utils.config import DATA_DIR, VIDEO_DIR, OUTPUT_DIR, ...
"""
import os
from pathlib import Path

# ---- Project root (2 levels up from this file: src/utils/config.py → TaiChi/) ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---- Data directories (keep existing dy_file/ layout intact) ----
DATA_DIR = PROJECT_ROOT / "dy_file"
VIDEO_DIR = DATA_DIR / "dy_video"
VIDEO_BASE64_DIR = DATA_DIR / "dy_video_base64"
VIDEO_CSV_DIR = DATA_DIR / "dy_video_csv"
VIDEO_CSV_SUB_DIR = VIDEO_CSV_DIR / "video_csv"
COMMENT_RAW_DIR = DATA_DIR / "dy_vloger_comment"
SPLIT_DATA_DIR = DATA_DIR / "split_all_data"
SPLIT_COMMENTS_DIR = DATA_DIR / "split_all_comments"
SPLIT_COMMENTS_LABELED_DIR = DATA_DIR / "split_all_comments_labeled"

# ---- Output directories ----
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_COMMENT_DIR = OUTPUT_DIR / "figures_comment"
FIGURES_VIDEO_DIR = OUTPUT_DIR / "figures_video"
FIGURES_FUSION_DIR = OUTPUT_DIR / "figures_fusion"
HEATMAP_DIR = OUTPUT_DIR / "heatmap"

# Ensure output directories exist
for _d in [FIGURES_COMMENT_DIR, FIGURES_VIDEO_DIR, FIGURES_FUSION_DIR, HEATMAP_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---- Key input CSV paths ----
COMMENT_LABELED_CLEANED_ONEHOT = VIDEO_CSV_DIR / "merged_dy_data_Allcomments_labeled_cleaned_onehot.csv"
COMMENT_LABELED_CLEANED = VIDEO_CSV_DIR / "merged_dy_data_Allcomments_labeled_cleaned.csv"
COMMENT_LABELED = VIDEO_CSV_DIR / "merged_dy_data_Allcomments_labeled.csv"
COMMENT_ALL = VIDEO_CSV_DIR / "merged_dy_data_Allcomments.csv"
VIDEO_MERGED = VIDEO_CSV_DIR / "merged_dy_data.csv"
VIDEO_MERGED_WITH_LABELS = VIDEO_CSV_SUB_DIR / "merged_dy_ALLData_with_labels.csv"
VIDEO_SEARCH = VIDEO_CSV_DIR / "merged_dy_search.csv"
COMMENT_MERGED = VIDEO_CSV_DIR / "merged_dy_data3_comments.csv"

# ---- FFmpeg paths ----
FFMPEG = r"C:\ProgramData\anaconda3\Library\bin\ffmpeg.exe"
FFPROBE = r"C:\ProgramData\anaconda3\Library\bin\ffprobe.exe"
