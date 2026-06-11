"""
=============================================================================
Nature-quality figures: Tai Chi Video Content Analysis
Arguments 1 & 2 — Supply-side production diversity & communication circles
=============================================================================
Input: merged_dy_ALLData_with_labels.csv (10,503 videos × 76 cols)
Output: dy_file/dy_video_csv/figures_video/ — SVG + PDF + TIFF per figure
Backend: Python (matplotlib + seaborn)
=============================================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import VIDEO_MERGED_WITH_LABELS, FIGURES_VIDEO_DIR

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. Paths & global config
# ============================================================
INPUT_PATH = str(VIDEO_MERGED_WITH_LABELS)
OUTPUT_DIR = FIGURES_VIDEO_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANG = 'zh'

# ----------------------------------------------------------
# Nature-style rcParams
# ----------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "legend.frameon": False,
    "legend.fontsize": 6.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})

# ============================================================
# 1. Colour palettes
# ============================================================
C_BAR_MAIN = '#5d8aa8'
C_HEATMAP = sns.color_palette("YlGnBu", 256)

# Style colors
C_STYLE = {
    '陈式': '#c44e52', '杨式': '#4c72b0', '吴式': '#55a868',
    '武式': '#dd8452', '孙式': '#937860', '武当太极': '#64b5cd',
    '24式': '#da8bc3', '其他': '#8c8c8c',
}

# Purpose colors
C_PURPOSE = {
    '教学': '#4c72b0', '养生': '#55a868', '日常': '#dd8452',
    '文化哲学': '#937860', '门派': '#64b5cd', '实战': '#c44e52',
    '赛事': '#da8bc3', '其他': '#8c8c8c',
}

# Scene colors
C_SCENE = {
    '山水自然': '#55a868', '古建筑': '#937860', '武馆': '#c44e52',
    '城市街道': '#4c72b0', '居家': '#dd8452', '寺庙道观': '#da8bc3',
    '校园': '#64b5cd', '其他': '#8c8c8c',
}

# Presenter colors
C_PRESENTER = {
    '专业武术家/道长': '#c44e52', '泛娱乐博主': '#4c72b0', '非科班出身': '#55a868',
}

# Style attribute colors (continuous-like)
C_ATTR = {
    '偏重柔和': '#55a868', '刚柔并济': '#4c72b0', '偏重刚猛': '#c44e52',
}

TOTAL_VIDEOS = None  # will be set after loading


# ============================================================
# 2. Data loading
# ============================================================
print('=' * 60)
print('Step 1: Loading video data...')
print('=' * 60)

df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig', low_memory=False)
TOTAL_VIDEOS = len(df)
print(f'Total videos: {TOTAL_VIDEOS:,}')

# Parse time
df['发布时间_dt'] = pd.to_datetime(df['发布时间'], errors='coerce')
df['年月'] = df['发布时间_dt'].dt.strftime('%Y-%m')

# Engagement metrics
ENGAGEMENT_COLS = ['点赞数', '评论数', '收藏数', '推荐数', '转发数']

# Video duration in seconds (原始单位是ms)
df['视频时长_秒'] = df['视频时长'] / 1000

print(f'Time range: {df["发布时间_dt"].min()} ~ {df["发布时间_dt"].max()}')
print(f'Total likes: {int(df["点赞数"].sum()):,}')
print(f'Total comments: {int(df["评论数"].sum()):,}')
print(f'Total saves: {int(df["收藏数"].sum()):,}')
print(f'Total shares: {int(df["转发数"].sum()):,}')

# ============================================================
# 3. Column group definitions
# ============================================================
DISPLAY_COLS = ['全身展示', '局部展示']
NARRATION_COLS = ['讲解类', '非讲解类']
LANGUAGE_COLS = ['中文', '英文', '双语']
GENDER_COLS = ['男性', '女性', '群体混合']
AGE_COLS = ['儿童', '青年', '中年', '老年']
PRESENTER_COLS = ['专业武术家/道长', '泛娱乐博主', '非科班出身']
STYLE_COLS_MAIN = ['陈式', '杨式', '吴式', '武式', '孙式', '武当太极', '24式']
STYLE_COLS_ALL = STYLE_COLS_MAIN + ['其他']
STYLE_ATTR_COLS = ['偏重柔和', '刚柔并济', '偏重刚猛']
PURPOSE_COLS_MAIN = ['赛事', '教学', '养生', '门派', '日常', '实战', '文化哲学']
PURPOSE_COLS_ALL = PURPOSE_COLS_MAIN + ['其他.1']
EFFECT_COLS_MAIN = ['无特效', '有特效', '古风慢摇', '节奏卡点']
EFFECT_COLS_ALL = EFFECT_COLS_MAIN + ['其他.2']
SCENE_COLS_MAIN = ['寺庙道观', '武馆', '山水自然', '城市街道', '居家', '校园', '古建筑']
SCENE_COLS_ALL = SCENE_COLS_MAIN + ['其他.3']
CLOTHING_COLS_MAIN = ['便服装', '太极服装', '古风服装', '现代潮流服装']
CLOTHING_COLS_ALL = CLOTHING_COLS_MAIN + ['其他.4']
MUSIC_INSTRUMENT_COLS = ['东方传统乐器', '西方乐器', '电子合成器', '白噪声环境音']
MUSIC_STYLE_COLS_MAIN = ['古风', '国风潮流', '戏曲', '原声', '旁白解说', '舒缓空灵', '大气磅礴', '动感活力', '网红BGM']
MUSIC_STYLE_COLS_ALL = MUSIC_STYLE_COLS_MAIN + ['其他.2']


def safe_sum(col):
    """Get sum of a one-hot column safely."""
    if col in df.columns:
        return int(df[col].sum())
    return 0


def build_series(cols):
    """Build a sorted Series from one-hot column sums (ascending for barh)."""
    data = {}
    for c in cols:
        if c in df.columns:
            data[c] = int(df[c].sum())
    return pd.Series(data).sort_values(ascending=True)


def build_cross_table(cols_a, cols_b, normalize='rows'):
    """Build crosstab DataFrame from two lists of one-hot columns."""
    rows, cols_labels = [], []
    mat_data = []
    for ca in cols_a:
        if ca not in df.columns:
            continue
        rows.append(ca)
        row_data = []
        for cb in cols_b:
            if cb not in df.columns:
                continue
            if not cols_labels or len(cols_labels) < len(cols_b):
                cols_labels.append(cb)
            mask = (df[ca] == 1) & (df[cb] == 1)
            row_data.append(int(mask.sum()))
        mat_data.append(row_data)

    result = pd.DataFrame(mat_data, index=rows, columns=cols_labels)
    if normalize == 'rows':
        result = result.div(result.sum(axis=1), axis=0) * 100
    elif normalize == 'cols':
        result = result.div(result.sum(axis=0), axis=1) * 100
    return result


def engagement_by_category(cat_cols):
    """Calculate average engagement metrics for each category."""
    results = {}
    for c in cat_cols:
        if c not in df.columns:
            continue
        mask = df[c] == 1
        subset = df[mask]
        n = len(subset)
        if n == 0:
            continue
        results[c] = {
            'count': n,
            'avg_likes': subset['点赞数'].mean(),
            'avg_comments': subset['评论数'].mean(),
            'avg_saves': subset['收藏数'].mean(),
            'avg_shares': subset['转发数'].mean(),
            'median_likes': subset['点赞数'].median(),
            'total_likes': int(subset['点赞数'].sum()),
        }
    return results


# ============================================================
# 4. Export helper
# ============================================================
def save_figure(fig, filename_stem):
    """Save figure as SVG + PDF + high-DPI TIFF."""
    base = OUTPUT_DIR / filename_stem
    fig.savefig(str(base) + '.svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(str(base) + '.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(str(base) + '.tiff', dpi=600, bbox_inches='tight', facecolor='white',
                edgecolor='none', pil_kwargs={'compression': 'tiff_lzw'})
    print(f'  Saved: {filename_stem}{{.svg,.pdf,.tiff}}')


# ============================================================
# 5. FIGURE: Argument 1 — Diverse Cognition & Value Identity (Supply Side)
# ============================================================

# --- Fig V1a: Tai Chi Style distribution ---
def fig_v1a_style_bar():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    s = build_series(STYLE_COLS_MAIN)
    colors = [C_STYLE.get(k, C_BAR_MAIN) for k in s.index]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.62)
    max_v = s.values.max()
    for bar, val in zip(bars, s.values):
        pct = val / TOTAL_VIDEOS * 100
        ax.text(bar.get_width() + max_v * 0.018, bar.get_y() + bar.get_height() / 2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=6, color='#334155')
    ax.set_xlim(0, max_v * 1.28)
    ax.set_xlabel('视频数量', fontsize=7, color='#475569')
    ax.set_title('(a) 太极拳流派视频分布', fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v1a_style_bar')
    plt.close(fig)


# --- Fig V1b: Purpose/Intent distribution ---
def fig_v1b_purpose_bar():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    s = build_series(PURPOSE_COLS_MAIN)
    colors = [C_PURPOSE.get(k, C_BAR_MAIN) for k in s.index]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.62)
    max_v = s.values.max()
    for bar, val in zip(bars, s.values):
        pct = val / TOTAL_VIDEOS * 100
        ax.text(bar.get_width() + max_v * 0.018, bar.get_y() + bar.get_height() / 2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=6, color='#334155')
    ax.set_xlim(0, max_v * 1.28)
    ax.set_xlabel('视频数量', fontsize=7, color='#475569')
    ax.set_title('(b) 视频意图/用途分布', fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v1b_purpose_bar')
    plt.close(fig)


# --- Fig V1c: Style × Purpose heatmap ---
def fig_v1c_style_purpose_heatmap():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    df_ct = build_cross_table(STYLE_COLS_MAIN, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 25},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('(c) 流派 × 意图（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('太极流派', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v1c_style_purpose_heatmap')
    plt.close(fig)


# --- Fig V1d: Style Attribute × Purpose ---
def fig_v1d_style_attr_purpose_heatmap():
    fig, ax = plt.subplots(figsize=(9.5, 2.2))
    df_ct = build_cross_table(STYLE_ATTR_COLS, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 20},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('(d) 风格属性 × 意图（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('风格属性', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v1d_attr_purpose_heatmap')
    plt.close(fig)


# --- Fig V1e: Presenter type distribution ---
def fig_v1e_presenter_purpose():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4),
                                    gridspec_kw={'width_ratios': [0.9, 2.4]})

    # Left: Presenter pie/donut
    s_pres = build_series(PRESENTER_COLS)
    colors_pres = [C_PRESENTER.get(k, '#8c8c8c') for k in s_pres.index]
    wedges, _ = ax1.pie(
        s_pres.values, labels=None, startangle=90,
        colors=colors_pres,
        wedgeprops=dict(width=0.38, edgecolor='white', linewidth=2),
    )
    ax1.text(0, 0, f'{TOTAL_VIDEOS:,}', ha='center', va='center',
             fontsize=13, fontweight='bold', color='#2c3e50')
    ax1.text(0, -0.15, '视频总数', ha='center', va='center', fontsize=6.5, color='#64748b')
    legend_labels = [f'{k} ({v:,}, {v/TOTAL_VIDEOS*100:.1f}%)' for k, v in zip(s_pres.index, s_pres.values)]
    ax1.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5),
               fontsize=6, frameon=False, handlelength=1.2, handleheight=1.2)
    ax1.set_title('博主类型分布', fontsize=8, fontweight='bold', pad=10)

    # Right: Presenter × Purpose heatmap
    df_ct = build_cross_table(PRESENTER_COLS, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax2,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 20},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax2.set_title('博主类型 × 视频意图（行百分比）', fontsize=8, fontweight='bold', pad=10)
    ax2.set_xlabel('视频意图', fontsize=7)
    ax2.set_ylabel('')
    ax2.tick_params(labelsize=6.5)
    for label in ax2.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax2.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)

    fig.suptitle('(e) 视频创作者类型与内容意图', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_v1e_presenter_purpose')
    plt.close(fig)


# --- Fig V1f: Production aesthetics — Scene × Clothing ---
def fig_v1f_scene_clothing_heatmap():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    df_ct = build_cross_table(SCENE_COLS_MAIN, CLOTHING_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 25},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('(f) 场景 × 服装（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('服装类型', fontsize=7)
    ax.set_ylabel('拍摄场景', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v1f_scene_clothing_heatmap')
    plt.close(fig)


# ============================================================
# 6. FIGURE: Argument 2 — Content Circles & Heterogeneity
# ============================================================

# --- Fig V2a: Scene distribution ---
def fig_v2a_scene_bar():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    s = build_series(SCENE_COLS_MAIN)
    colors = [C_SCENE.get(k, C_BAR_MAIN) for k in s.index]
    bars = ax.barh(s.index, s.values, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.62)
    max_v = s.values.max()
    for bar, val in zip(bars, s.values):
        pct = val / TOTAL_VIDEOS * 100
        ax.text(bar.get_width() + max_v * 0.018, bar.get_y() + bar.get_height() / 2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=6, color='#334155')
    ax.set_xlim(0, max_v * 1.28)
    ax.set_xlabel('视频数量', fontsize=7, color='#475569')
    ax.set_title('(a) 拍摄场景分布', fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v2a_scene_bar')
    plt.close(fig)


# --- Fig V2b: Music style distribution ---
def fig_v2b_music_bar():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    s = build_series(MUSIC_STYLE_COLS_MAIN)
    cmap_music = sns.color_palette("crest", n_colors=len(s))
    bars = ax.barh(s.index, s.values, color=cmap_music, edgecolor='white',
                   linewidth=0.8, height=0.62)
    max_v = s.values.max()
    for bar, val in zip(bars, s.values):
        pct = val / TOTAL_VIDEOS * 100
        ax.text(bar.get_width() + max_v * 0.018, bar.get_y() + bar.get_height() / 2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=6, color='#334155')
    ax.set_xlim(0, max_v * 1.28)
    ax.set_xlabel('视频数量', fontsize=7, color='#475569')
    ax.set_title('(b) 音乐风格分布', fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v2b_music_bar')
    plt.close(fig)


# --- Fig V2c: Scene × Purpose heatmap ---
def fig_v2c_scene_purpose_heatmap():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    df_ct = build_cross_table(SCENE_COLS_MAIN, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 25},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('(c) 场景 × 意图（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('拍摄场景', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v2c_scene_purpose_heatmap')
    plt.close(fig)


# --- Fig V2d: Music × Style heatmap ---
def fig_v2d_music_style_heatmap():
    # Select top music styles for readability
    music_s = build_series(MUSIC_STYLE_COLS_MAIN)
    top_music = list(music_s.nlargest(7).index)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    df_ct = build_cross_table(top_music, STYLE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 25},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('(d) 音乐风格 × 太极流派（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('太极流派', fontsize=7)
    ax.set_ylabel('音乐风格', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v2d_music_style_heatmap')
    plt.close(fig)


# --- Fig V2e: Gender × Age distribution for different styles ---
def fig_v2e_demographics():
    """Show how different styles appeal to different demographics (gender/age of presenters)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Gender × Style
    df_gen = build_cross_table(STYLE_COLS_MAIN, GENDER_COLS, normalize='rows')
    sns.heatmap(
        df_gen, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax1,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 20},
        vmin=0, vmax=df_gen.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax1.set_title('流派 × 性别（行百分比）', fontsize=8, fontweight='bold', pad=10)
    ax1.set_xlabel('性别', fontsize=7)
    ax1.set_ylabel('太极流派', fontsize=7)
    ax1.tick_params(labelsize=6.5)

    # Age × Style
    df_age = build_cross_table(STYLE_COLS_MAIN, AGE_COLS, normalize='rows')
    sns.heatmap(
        df_age, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax2,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 20},
        vmin=0, vmax=df_age.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax2.set_title('流派 × 年龄（行百分比）', fontsize=8, fontweight='bold', pad=10)
    ax2.set_xlabel('年龄段', fontsize=7)
    ax2.set_ylabel('')
    ax2.tick_params(labelsize=6.5)

    for ax_i in [ax1, ax2]:
        cbar = ax_i.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5.5)
        cbar.ax.yaxis.label.set_fontsize(6.5)

    fig.suptitle('(e) 各流派视频中的博主人口特征', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_v2e_demographics')
    plt.close(fig)


# --- Fig V2f: Content circles — Style × Scene × Purpose multi-panel ---
def fig_v2f_content_circles():
    """Multi-panel figure showing content 'circles' — how styles cluster with
    specific scenes and purposes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # Panel 1: Style × Language
    ax = axes[0]
    df_lang = build_cross_table(STYLE_COLS_MAIN, LANGUAGE_COLS, normalize='rows')
    sns.heatmap(
        df_lang, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': '%'},
        vmin=0, annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title('流派 × 语言', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Panel 2: Purpose × Effect
    ax = axes[1]
    df_eff = build_cross_table(PURPOSE_COLS_MAIN, EFFECT_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_eff, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': '%'},
        vmin=0, annot_kws={'fontsize': 5.5, 'fontweight': 'bold'},
    )
    ax.set_title('意图 × 特效', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha('right')

    # Panel 3: Scene × Music Instrument
    ax = axes[2]
    df_mi = build_cross_table(SCENE_COLS_MAIN, MUSIC_INSTRUMENT_COLS, normalize='rows')
    sns.heatmap(
        df_mi, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': '%'},
        vmin=0, annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title('场景 × 乐器类型', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Panel 4: Style × Narration type
    ax = axes[3]
    df_narr = build_cross_table(STYLE_COLS_MAIN, NARRATION_COLS, normalize='rows')
    sns.heatmap(
        df_narr, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': '%'},
        vmin=0, annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title('流派 × 讲解方式', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Panel 5: Purpose × Display type
    ax = axes[4]
    df_disp = build_cross_table(PURPOSE_COLS_MAIN, DISPLAY_COLS, normalize='rows')
    sns.heatmap(
        df_disp, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': '%'},
        vmin=0, annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title('意图 × 展示方式', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Panel 6: Purpose × Music (top music)
    ax = axes[5]
    music_s = build_series(MUSIC_STYLE_COLS_MAIN)
    top_m6 = list(music_s.nlargest(6).index)
    df_pm = build_cross_table(PURPOSE_COLS_MAIN, top_m6, normalize='rows')
    sns.heatmap(
        df_pm, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.6, 'label': '%'},
        vmin=0, annot_kws={'fontsize': 5.5, 'fontweight': 'bold'},
    )
    ax.set_title('意图 × 音乐风格（Top 6）', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha('right')

    # Clean up colorbars
    for ax_i in axes:
        if hasattr(ax_i, 'collections') and ax_i.collections:
            cbar = ax_i.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=5)
                cbar.ax.yaxis.label.set_fontsize(6)

    fig.suptitle('(f) 太极视频内容圈层：多维交叉分析',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(pad=3)
    save_figure(fig, 'fig_v2f_content_circles')
    plt.close(fig)


# ============================================================
# 7. FIGURE: Temporal & Engagement Analysis
# ============================================================

# --- Fig V3a: Monthly video posting volume ---
def fig_v3a_temporal_trend():
    monthly = df.groupby('年月').size().sort_index()
    # Filter to reasonable range
    monthly = monthly[monthly.index >= '2019-10']
    monthly = monthly[monthly.index <= '2026-04']

    # Monthly likes
    monthly_likes = df.groupby('年月')['点赞数'].sum().sort_index()
    monthly_likes = monthly_likes[monthly_likes.index >= '2019-10']
    monthly_likes = monthly_likes[monthly_likes.index <= '2026-04']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                    gridspec_kw={'height_ratios': [1.5, 1]})

    months = list(monthly.index)
    x = range(len(months))

    # Top: bar chart with gradient
    norm = mpl.colors.Normalize(vmin=min(monthly.values), vmax=max(monthly.values))
    cmap_grad = mpl.colors.LinearSegmentedColormap.from_list(
        'blue_grad', ['#B4C0E4', '#7884B4', '#3775BA', '#0F4D92'])
    bar_colors = [cmap_grad(norm(v)) for v in monthly.values]
    ax1.bar(x, monthly.values, color=bar_colors, width=0.72,
            edgecolor='white', linewidth=0.3)
    ax1.set_ylabel('视频发布量', fontsize=7, color='#475569')
    ax1.set_title('(a) 视频月度发布量与获赞趋势', fontsize=9, fontweight='bold', pad=8)
    ax1.tick_params(labelsize=6)

    # Trend line
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, monthly_likes.values, color='#c44e52', linewidth=1.2, alpha=0.7, marker='o', markersize=2)
    ax1_twin.set_ylabel('总获赞量', fontsize=7, color='#c44e52')
    ax1_twin.tick_params(labelsize=6, colors='#c44e52')
    ax1_twin.spines['right'].set_visible(True)
    ax1_twin.spines['right'].set_color('#c44e52')

    # Bottom: cumulative videos
    cumsum = monthly.cumsum()
    ax2.fill_between(x, 0, cumsum.values, color='#4c72b0', alpha=0.35,
                     edgecolor='#4c72b0', linewidth=0.8)
    ax2.plot(x, cumsum.values, color='#4c72b0', linewidth=1.2)
    ax2.set_ylabel('累计视频量', fontsize=7, color='#475569')
    ax2.set_xlabel('月份', fontsize=7, color='#475569')

    # X-axis labels
    step = max(1, len(months) // 16)
    tick_pos = list(range(0, len(months), step))
    tick_labels = [months[i] for i in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=5.5)
    ax2.tick_params(labelsize=6)

    # Panel labels
    for ax_i, lbl in zip((ax1, ax2), ('a', 'b')):
        ax_i.text(-0.04, 1.04, lbl, transform=ax_i.transAxes,
                  fontsize=10, fontweight='bold', color='#272727',
                  ha='left', va='bottom')

    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v3a_temporal_trend')
    plt.close(fig)


# --- Fig V3b: Engagement by content type ---
def fig_v3b_engagement_by_category():
    """Average likes and engagement rates by style and purpose."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    # Panel 1: Avg likes by style
    ax = axes[0]
    eng_style = engagement_by_category(STYLE_COLS_MAIN)
    styles_ordered = sorted(eng_style.items(), key=lambda x: -x[1]['avg_likes'])
    names = [s[0] for s in styles_ordered]
    vals = [s[1]['avg_likes'] for s in styles_ordered]
    colors = [C_STYLE.get(n, C_BAR_MAIN) for n in names]
    bars = ax.barh(names, vals, color=colors, edgecolor='white',
                   linewidth=0.6, height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:,.0f}', va='center', fontsize=5.5, color='#334155')
    ax.set_title('流派 × 平均点赞', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('平均点赞数', fontsize=6.5, color='#475569')
    ax.tick_params(labelsize=6)
    ax.set_xlim(0, max(vals) * 1.2)

    # Panel 2: Avg likes by purpose
    ax = axes[1]
    eng_purp = engagement_by_category(PURPOSE_COLS_MAIN)
    purps_ordered = sorted(eng_purp.items(), key=lambda x: -x[1]['avg_likes'])
    names = [s[0] for s in purps_ordered]
    vals = [s[1]['avg_likes'] for s in purps_ordered]
    colors = [C_PURPOSE.get(n, C_BAR_MAIN) for n in names]
    bars = ax.barh(names, vals, color=colors, edgecolor='white',
                   linewidth=0.6, height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:,.0f}', va='center', fontsize=5.5, color='#334155')
    ax.set_title('意图 × 平均点赞', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('平均点赞数', fontsize=6.5, color='#475569')
    ax.tick_params(labelsize=6)
    ax.set_xlim(0, max(vals) * 1.2)

    # Panel 3: Engagement rate (likes/views proxy: likes+duration interaction)
    ax = axes[2]
    # Calculate "engagement efficiency" — median likes per purpose
    eng_data = {}
    for c in PURPOSE_COLS_MAIN + STYLE_COLS_MAIN:
        if c not in df.columns:
            continue
        mask = df[c] == 1
        subset = df[mask]
        if len(subset) > 20:
            # Use log-transformed likes to handle skew
            eng_data[c] = {
                'median_likes': subset['点赞数'].median(),
                'n': len(subset),
                'category': '意图' if c in PURPOSE_COLS_MAIN else '流派'
            }

    # Show median likes for key categories
    top_by_median = sorted(eng_data.items(), key=lambda x: -x[1]['median_likes'])[:12]
    names_med = [t[0] for t in top_by_median]
    vals_med = [t[1]['median_likes'] for t in top_by_median]
    colors_med = [C_PURPOSE.get(n, C_STYLE.get(n, C_BAR_MAIN)) for n in names_med]
    bars = ax.barh(names_med, vals_med, color=colors_med, edgecolor='white',
                   linewidth=0.6, height=0.6)
    for bar, val in zip(bars, vals_med):
        ax.text(bar.get_width() + max(vals_med) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:,.0f}', va='center', fontsize=5.5, color='#334155')
    ax.set_title('中位数点赞（Top 12）', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('中位数点赞', fontsize=6.5, color='#475569')
    ax.tick_params(labelsize=6)
    ax.set_xlim(0, max(vals_med) * 1.2)

    fig.suptitle('(b) 内容类型 × 传播效果', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_v3b_engagement_category')
    plt.close(fig)


# --- Fig V3c: Style attribute × Scene — where different styles are practiced ---
def fig_v3c_style_attr_scene():
    fig, ax = plt.subplots(figsize=(9.5, 2.2))
    df_ct = build_cross_table(STYLE_ATTR_COLS, SCENE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 20},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('(c) 风格属性 × 拍摄场景（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('拍摄场景', fontsize=7)
    ax.set_ylabel('风格属性', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v3c_style_attr_scene')
    plt.close(fig)


# --- Fig V3d: Clothing × Purpose ---
def fig_v3d_clothing_purpose():
    fig, ax = plt.subplots(figsize=(9.5, 3))
    df_ct = build_cross_table(CLOTHING_COLS_MAIN, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 22},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('(d) 服装 × 意图（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('服装类型', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v3d_clothing_purpose')
    plt.close(fig)


# --- Fig V3e: Effect × Purpose ---
def fig_v3e_effect_purpose():
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    df_ct = build_cross_table(EFFECT_COLS_MAIN, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ct, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 22},
        vmin=0, vmax=df_ct.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title('(e) 特效 × 意图（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('特效类型', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v3e_effect_purpose')
    plt.close(fig)


# --- Fig V3f: Comprehensive multi-panel overview ---
def fig_v3f_overview_dashboard():
    """4-panel dashboard: age × purpose, gender × purpose, music instrument × purpose, style attribute × clothing."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Panel 1: Age × Purpose
    ax = axes[0, 0]
    df_ap = build_cross_table(AGE_COLS, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(df_ap, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.55, 'label': '占比 (%)'},
                vmin=0, annot_kws={'fontsize': 6, 'fontweight': 'bold'})
    ax.set_title('年龄段 × 意图', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Panel 2: Gender × Purpose
    ax = axes[0, 1]
    df_gp = build_cross_table(GENDER_COLS, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(df_gp, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.55, 'label': '占比 (%)'},
                vmin=0, annot_kws={'fontsize': 6, 'fontweight': 'bold'})
    ax.set_title('性别 × 意图', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Panel 3: Music Instrument × Purpose
    ax = axes[1, 0]
    df_mp = build_cross_table(MUSIC_INSTRUMENT_COLS, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(df_mp, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.55, 'label': '占比 (%)'},
                vmin=0, annot_kws={'fontsize': 6, 'fontweight': 'bold'})
    ax.set_title('乐器类型 × 意图', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Panel 4: Style Attribute × Clothing
    ax = axes[1, 1]
    df_sc = build_cross_table(STYLE_ATTR_COLS, CLOTHING_COLS_MAIN, normalize='rows')
    sns.heatmap(df_sc, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.55, 'label': '占比 (%)'},
                vmin=0, annot_kws={'fontsize': 6, 'fontweight': 'bold'})
    ax.set_title('风格属性 × 服装', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    for ax_i in axes.flatten():
        if ax_i.collections:
            cbar = ax_i.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=5)
                cbar.ax.yaxis.label.set_fontsize(6)
        for label in ax_i.get_xticklabels():
            label.set_rotation(20)
            label.set_ha('right')

    fig.suptitle('(f) 视频内容与人口特征多维交叉',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(pad=3)
    save_figure(fig, 'fig_v3f_demographic_cross')
    plt.close(fig)


# ============================================================
# 8. Summary statistics export
# ============================================================
def export_summary():
    lines = []
    lines.append('=' * 70)
    lines.append('Tai Chi Video Content Analysis — Summary')
    lines.append('=' * 70)
    lines.append(f'Total videos: {TOTAL_VIDEOS:,}')
    lines.append(f'Total likes: {int(df["点赞数"].sum()):,}')
    lines.append(f'Total comments: {int(df["评论数"].sum()):,}')
    lines.append(f'Total saves: {int(df["收藏数"].sum()):,}')
    lines.append(f'Total shares: {int(df["转发数"].sum()):,}')
    lines.append(f'Time range: {df["发布时间_dt"].min()} ~ {df["发布时间_dt"].max()}')
    lines.append('')

    lines.append('--- Tai Chi Styles ---')
    for c in STYLE_COLS_ALL:
        val = safe_sum(c)
        pct = val / TOTAL_VIDEOS * 100
        lines.append(f'  {c}: {val:,} ({pct:.1f}%)')
    lines.append('')

    lines.append('--- Purposes/Intents ---')
    for c in PURPOSE_COLS_ALL:
        val = safe_sum(c)
        pct = val / TOTAL_VIDEOS * 100
        lines.append(f'  {c}: {val:,} ({pct:.1f}%)')
    lines.append('')

    lines.append('--- Style Attributes ---')
    for c in STYLE_ATTR_COLS:
        val = safe_sum(c)
        pct = val / TOTAL_VIDEOS * 100
        lines.append(f'  {c}: {val:,} ({pct:.1f}%)')
    lines.append('')

    lines.append('--- Presenter Types ---')
    for c in PRESENTER_COLS:
        val = safe_sum(c)
        pct = val / TOTAL_VIDEOS * 100
        lines.append(f'  {c}: {val:,} ({pct:.1f}%)')
    lines.append('')

    lines.append('--- Scenes ---')
    for c in SCENE_COLS_ALL:
        val = safe_sum(c)
        pct = val / TOTAL_VIDEOS * 100
        lines.append(f'  {c}: {val:,} ({pct:.1f}%)')
    lines.append('')

    lines.append('--- Clothing ---')
    for c in CLOTHING_COLS_ALL:
        val = safe_sum(c)
        pct = val / TOTAL_VIDEOS * 100
        lines.append(f'  {c}: {val:,} ({pct:.1f}%)')
    lines.append('')

    lines.append('--- Music Styles ---')
    for c in MUSIC_STYLE_COLS_ALL:
        if c in df.columns:
            val = int(df[c].sum())
            pct = val / TOTAL_VIDEOS * 100
            lines.append(f'  {c}: {val:,} ({pct:.1f}%)')
    lines.append('')

    # Top videos
    lines.append('--- Top 10 Most Liked Videos ---')
    top10 = df.nlargest(10, '点赞数')
    for i, (_, row) in enumerate(top10.iterrows()):
        lines.append(f'  {i+1}. likes={int(row["点赞数"]):,}, comments={int(row["评论数"]):,}, '
                     f'saves={int(row["收藏数"]):,}, shares={int(row["转发数"]):,}')
    lines.append('')

    # Engagement by dimension
    lines.append('--- Avg Likes by Style ---')
    for c in STYLE_COLS_MAIN:
        if c in df.columns:
            mask = df[c] == 1
            avg_l = df.loc[mask, '点赞数'].mean()
            med_l = df.loc[mask, '点赞数'].median()
            lines.append(f'  {c}: avg={avg_l:,.0f}, median={med_l:,.0f}, n={mask.sum():,}')
    lines.append('')

    lines.append('--- Avg Likes by Purpose ---')
    for c in PURPOSE_COLS_MAIN:
        if c in df.columns:
            mask = df[c] == 1
            avg_l = df.loc[mask, '点赞数'].mean()
            med_l = df.loc[mask, '点赞数'].median()
            lines.append(f'  {c}: avg={avg_l:,.0f}, median={med_l:,.0f}, n={mask.sum():,}')

    path = OUTPUT_DIR / 'summary_statistics_video.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  Saved: summary_statistics_video.txt')


# ============================================================
# 9. Main
# ============================================================
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('Video Content Analysis — Nature-Quality Figure Generation')
    print(f'Language: {LANG.upper()}  |  Output: {OUTPUT_DIR}')
    print(f'Total videos: {TOTAL_VIDEOS:,}')
    print('=' * 60)

    print('\n--- Argument 1: Diverse Cognition & Value Identity (Supply Side) ---')
    fig_v1a_style_bar()
    fig_v1b_purpose_bar()
    fig_v1c_style_purpose_heatmap()
    fig_v1d_style_attr_purpose_heatmap()
    fig_v1e_presenter_purpose()
    fig_v1f_scene_clothing_heatmap()

    print('\n--- Argument 2: Content Circles & Heterogeneity ---')
    fig_v2a_scene_bar()
    fig_v2b_music_bar()
    fig_v2c_scene_purpose_heatmap()
    fig_v2d_music_style_heatmap()
    fig_v2e_demographics()
    fig_v2f_content_circles()

    print('\n--- Temporal & Engagement Analysis ---')
    fig_v3a_temporal_trend()
    fig_v3b_engagement_by_category()
    fig_v3c_style_attr_scene()
    fig_v3d_clothing_purpose()
    fig_v3e_effect_purpose()
    fig_v3f_overview_dashboard()

    print('\n--- Summary Statistics ---')
    export_summary()

    print('\n' + '=' * 60)
    print(f'Done. All figures saved to: {OUTPUT_DIR}')
    print(f'Total: 18 figures generated')
    print('=' * 60)
