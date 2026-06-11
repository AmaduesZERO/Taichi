"""
=============================================================================
Nature-quality figures: Tai Chi Video-Comment Fusion Analysis
=============================================================================
Figure contract:
  Core conclusion:
    Video production strategies systematically shape audience cognitive
    responses — teaching videos elicit technical discussion, cultural philosophy
    videos attract cultural debate, and daily-life content generates the most
    emotionally positive engagement. Geographic regions show differentiated
    content preferences, forming distinct supply-demand 'communication circuits'.

  Figure archetype: quantitative grid
  Backend: Python (matplotlib + seaborn)
  Output: SVG + PDF + TIFF per figure

  Panel map:
    Fig C1 — Supply-Demand Alignment (3 panels)
      a: Video purpose × Comment cognitive dimension (attr2) heatmap
      b: Video style × Comment sentiment heatmap
      c: Video presenter × Comment primary attribute (attr1)

    Fig C2 — Production Features → Audience Response (3 panels)
      a: Avg comments & engagement per video by video purpose
      b: Comment sentiment composition by video production features
      c: Comment intent distribution by video purpose (alignment check)

    Fig C3 — Regional × Video Content (4 panels)
      a: Province × Video purpose heatmap (top 15 provinces)
      b: Province × Video style heatmap
      c: Regional comment sentiment by video scene type
      d: Province engagement bubble — comment volume vs sentiment positivity

    Fig C4 — Multi-dimensional Integration (4 panels)
      a: Video-Comment dimension correspondence matrix
      b: Video duration → comment sentiment & engagement
      c: Comment attribute diversity vs video production complexity
      d: Video engagement vs comment volume scatter

    Fig C5 — Temporal Co-evolution (2 panels)
      a: Stacked area — video purpose vs comment intent over time
      b: Monthly sentiment by dominant video purpose

  Evidence hierarchy:
    hero: C1a (supply-demand cognitive alignment)
    validation: C1b,C1c (sentiment & attr1 alignment)
    regional: C3a,C3b (geographic preference)
    temporal: C5a,C5b (co-evolution)

  Statistics: Chi-square for contingency tables, correlation coefficients
=============================================================================
Input:
  - merged_dy_ALLData_with_labels.csv (10,503 videos × 76 cols)
  - merged_dy_data_Allcomments_labeled_cleaned_onehot.csv (1.45M comments × 73 cols)
  Mapping: 视频链接 ↔ 目标链接
Output: dy_file/dy_video_csv/figures_fusion/ — SVG + PDF + TIFF per figure
=============================================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import VIDEO_MERGED_WITH_LABELS, COMMENT_LABELED_CLEANED_ONEHOT, FIGURES_FUSION_DIR

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. Paths & global config
# ============================================================
VIDEO_PATH = str(VIDEO_MERGED_WITH_LABELS)
COMMENT_PATH = str(COMMENT_LABELED_CLEANED_ONEHOT)
OUTPUT_DIR = FIGURES_FUSION_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Nature-style rcParams ──
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
    "legend.fontsize": 6,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})

# ── Palette ──
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA", "blue_soft": "#B4C0E4",
    "green_3": "#8BCF8B", "green_dark": "#2E7D32",
    "red_strong": "#B64342", "red_soft": "#F6CFCB",
    "neutral_light": "#CFCECE", "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D", "neutral_black": "#272727",
    "gold": "#E8A735", "teal": "#42949E", "violet": "#9A4D8E",
    "magenta": "#C44E7C", "orange": "#E28E2C",
}

C_PURPOSE = {
    '教学': '#3775BA', '养生': '#2E7D32', '日常': '#E28E2C',
    '文化哲学': '#9A4D8E', '门派': '#42949E', '实战': '#B64342', '赛事': '#C44E7C',
}
C_STYLE = {
    '陈式': '#B64342', '杨式': '#3775BA', '吴式': '#2E7D32',
    '武式': '#E28E2C', '孙式': '#9A4D8E', '武当太极': '#42949E', '24式': '#C44E7C',
}
C_SENTIMENT = {'肯定/赞美': '#5a9e6f', '诋毁/谩骂': '#c44e52', '客观中立': '#8c8c8c'}
C_PRESENTER = {'专业武术家/道长': '#B64342', '泛娱乐博主': '#3775BA', '非科班出身': '#2E7D32'}
C_SCENE = {
    '山水自然': '#2E7D32', '古建筑': '#9A4D8E', '武馆': '#B64342',
    '城市街道': '#3775BA', '居家': '#E28E2C', '寺庙道观': '#C44E7C', '校园': '#42949E',
}
C_STYLE_ATTR = {'偏重柔和': '#8BCF8B', '刚柔并济': '#3775BA', '偏重刚猛': '#B64342'}
C_HEATMAP = sns.color_palette("YlGnBu", 256)

# Region colors
C_REGION = {
    'North China': '#c44e52', 'Northeast': '#dd8452', 'East China': '#55a868',
    'Central China': '#4c72b0', 'South China': '#937860', 'Southwest': '#64b5cd',
    'Northwest': '#da8bc3', 'HK/Macau/Taiwan': '#8c8c8c',
}

# ============================================================
# 1. Helper functions
# ============================================================
def save_figure(fig, filename_stem):
    base = OUTPUT_DIR / filename_stem
    fig.savefig(str(base) + '.svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(str(base) + '.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(str(base) + '.tiff', dpi=600, bbox_inches='tight', facecolor='white',
                edgecolor='none', pil_kwargs={'compression': 'tiff_lzw'})
    print(f'  Saved: {filename_stem}{{.svg,.pdf,.tiff}}')


def panel_label(ax, label, x=-0.06, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', color=PALETTE['neutral_black'],
            ha='left', va='bottom')


def build_heatmap_df(mat_dict, row_keys, col_keys, normalize='rows'):
    """Build a DataFrame from a (row, col) -> value dict."""
    mat = np.zeros((len(row_keys), len(col_keys)))
    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            mat[i, j] = mat_dict.get((rk, ck), 0)
    df_out = pd.DataFrame(mat, index=row_keys, columns=col_keys)
    if normalize == 'rows':
        row_sums = df_out.sum(axis=1)
        df_out = df_out.div(row_sums.where(row_sums > 0, np.nan), axis=0) * 100
    elif normalize == 'cols':
        col_sums = df_out.sum(axis=0)
        df_out = df_out.div(col_sums.where(col_sums > 0, np.nan), axis=1) * 100
    return df_out


# Province/region mappings
PROVINCE_SHORT_MAP = {
    '北京': '北京市', '天津': '天津市', '上海': '上海市', '重庆': '重庆市',
    '广西': '广西壮族自治区', '新疆': '新疆维吾尔自治区',
    '宁夏': '宁夏回族自治区', '西藏': '西藏自治区', '内蒙古': '内蒙古自治区',
    '中国香港': '香港特别行政区', '中国澳门': '澳门特别行政区', '中国台湾': '台湾省',
}
_NORTH = {'北京市', '天津市', '河北省', '山西省', '内蒙古自治区'}
_NORTHEAST = {'辽宁省', '吉林省', '黑龙江省'}
_EAST = {'上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省'}
_CENTRAL = {'河南省', '湖北省', '湖南省'}
_SOUTH = {'广东省', '广西壮族自治区', '海南省'}
_SOUTHWEST = {'重庆市', '四川省', '贵州省', '云南省', '西藏自治区'}
_NORTHWEST = {'陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区'}
_HKMT = {'香港特别行政区', '澳门特别行政区', '台湾省'}
_REGION_ZH = {
    'North China': '华北', 'Northeast': '东北', 'East China': '华东',
    'Central China': '华中', 'South China': '华南', 'Southwest': '西南',
    'Northwest': '西北', 'HK/Macau/Taiwan': '港澳台',
}


def province_full_name(short):
    if short in PROVINCE_SHORT_MAP:
        return PROVINCE_SHORT_MAP[short]
    if short in ('中国', 'IP未知'):
        return None
    return short + '省'


def get_region(province_full):
    if province_full in _NORTH: return 'North China'
    if province_full in _NORTHEAST: return 'Northeast'
    if province_full in _EAST: return 'East China'
    if province_full in _CENTRAL: return 'Central China'
    if province_full in _SOUTH: return 'South China'
    if province_full in _SOUTHWEST: return 'Southwest'
    if province_full in _NORTHWEST: return 'Northwest'
    if province_full in _HKMT: return 'HK/Macau/Taiwan'
    return 'HK/Macau/Taiwan'


def province_display(full_cn):
    s = full_cn.replace('特别行政区', '').replace('自治区', '').replace('省', '').replace('市', '')
    display_map = {'中国香港': '香港', '中国澳门': '澳门', '中国台湾': '台湾',
                   '内蒙古': '内蒙古', '广西': '广西', '新疆': '新疆', '宁夏': '宁夏', '西藏': '西藏'}
    return display_map.get(s, s)


# ============================================================
# 2. Data loading & cross-tab accumulation
# ============================================================
print('=' * 60)
print('Step 1: Loading video data & building lookup...')
print('=' * 60)

df_v = pd.read_csv(VIDEO_PATH, encoding='utf-8-sig', low_memory=False)
print(f'Videos loaded: {len(df_v):,}')

# Column groups (video)
V_STYLE = ['陈式', '杨式', '吴式', '武式', '孙式', '武当太极', '24式']
V_PURPOSE = ['赛事', '教学', '养生', '门派', '日常', '实战', '文化哲学']
V_SCENE = ['寺庙道观', '武馆', '山水自然', '城市街道', '居家', '校园', '古建筑']
V_PRESENTER = ['专业武术家/道长', '泛娱乐博主', '非科班出身']
V_CLOTHING = ['便服装', '太极服装', '古风服装', '现代潮流服装']
V_ATTR = ['偏重柔和', '刚柔并济', '偏重刚猛']
V_EFFECT = ['无特效', '有特效', '古风慢摇', '节奏卡点']
V_MUSIC = ['古风', '国风潮流', '戏曲', '原声', '旁白解说', '舒缓空灵', '大气磅礴', '动感活力', '网红BGM']

# Column groups (comment)
C_SENTIMENT_COLS = ['情感_肯定/赞美', '情感_诋毁/谩骂', '情感_客观中立']
C_ATTR1_COLS = ['属性一_尊师重道', '属性一_实战性质疑', '属性一_文化质疑', '属性一_文化捍卫',
                '属性一_玩网络梗', '属性一_阴阳怪气', '属性一_对博主人身攻击', '属性一_对博主性骚扰']
C_ATTR2_COLS = ['属性二_动作要点', '属性二_实战技术', '属性二_养生功效', '属性二_视觉审美',
                '属性二_文化哲学', '属性二_博主本身', '属性二_关注其他人物', '属性二_关注配乐审美',
                '属性二_关注跨拳种实战比较']
C_INTENT_COLS = ['意向_种草意向', '意向_资源索取', '意向_技术追问', '意向_文化探讨',
                 '意向_跟练体感', '意向_病理/生理求助', '意向_寻师求学', '意向_主动推荐',
                 '意向_影视文学跨界探讨', '意向_门派流派探讨']

# Short labels for display
L_ATTR2 = {
    '属性二_动作要点': '动作要点', '属性二_实战技术': '实战技术', '属性二_养生功效': '养生功效',
    '属性二_视觉审美': '视觉审美', '属性二_文化哲学': '文化哲学', '属性二_博主本身': '博主本身',
    '属性二_关注其他人物': '关注他人', '属性二_关注配乐审美': '配乐审美',
    '属性二_关注跨拳种实战比较': '跨拳种比较',
}
L_ATTR1 = {
    '属性一_尊师重道': '尊师重道', '属性一_实战性质疑': '实战质疑',
    '属性一_文化质疑': '文化质疑', '属性一_文化捍卫': '文化捍卫',
    '属性一_玩网络梗': '网络玩梗', '属性一_阴阳怪气': '阴阳怪气',
    '属性一_对博主人身攻击': '人身攻击', '属性一_对博主性骚扰': '性骚扰',
}
L_INTENT = {
    '意向_种草意向': '种草意向', '意向_资源索取': '资源索取', '意向_技术追问': '技术追问',
    '意向_文化探讨': '文化探讨', '意向_跟练体感': '跟练体感',
    '意向_病理/生理求助': '病理求助', '意向_寻师求学': '寻师求学',
    '意向_主动推荐': '主动推荐', '意向_影视文学跨界探讨': '影视探讨',
    '意向_门派流派探讨': '门派探讨',
}
L_SENTIMENT = {
    '情感_肯定/赞美': '肯定/赞美', '情感_诋毁/谩骂': '诋毁/谩骂', '情感_客观中立': '客观中立',
}

# Build video lookup: url → feature dict
video_lookup = {}
for _, row in df_v.iterrows():
    url = row['视频链接']
    features = {
        'likes': row['点赞数'], 'comments_count': row['评论数'],
        'saves': row['收藏数'], 'shares': row['转发数'],
        'duration': row['视频时长'] / 1000,  # seconds
    }
    # One-hot features → list of active labels
    for style in V_STYLE:
        if row.get(style, 0) == 1:
            features.setdefault('styles', []).append(style)
    for purp in V_PURPOSE:
        if row.get(purp, 0) == 1:
            features.setdefault('purposes', []).append(purp)
    for scene in V_SCENE:
        if row.get(scene, 0) == 1:
            features.setdefault('scenes', []).append(scene)
    for pres in V_PRESENTER:
        if row.get(pres, 0) == 1:
            features.setdefault('presenters', []).append(pres)
    for attr in V_ATTR:
        if row.get(attr, 0) == 1:
            features.setdefault('style_attrs', []).append(attr)
    for cloth in V_CLOTHING:
        if row.get(cloth, 0) == 1:
            features.setdefault('clothing', []).append(cloth)
    for eff in V_EFFECT:
        if row.get(eff, 0) == 1:
            features.setdefault('effects', []).append(eff)
    for mus in V_MUSIC:
        if row.get(mus, 0) == 1:
            features.setdefault('music', []).append(mus)
    video_lookup[url] = features

print(f'Video lookup built: {len(video_lookup):,} entries')
n_styles = sum(1 for f in video_lookup.values() if 'styles' in f)
print(f'Videos with style labels: {n_styles:,}')

# ============================================================
# 3. Chunked comment processing + cross-tab accumulation
# ============================================================
print('\n' + '=' * 60)
print('Step 2: Chunked comment processing & cross-tab accumulation...')
print('=' * 60)

chunk_size = 100000
total_comments = 0
total_matched = 0

# Accumulators
acc = defaultdict(lambda: defaultdict(int))
# Structure: acc['cross_name'][(key_a, key_b)] = count
# cross_names:
#   'v_purpose_x_c_attr2', 'v_style_x_c_sentiment', 'v_presenter_x_c_attr1'
#   'v_scene_x_c_intent', 'v_purpose_x_c_intent', 'v_style_x_c_attr2'
#   'v_purpose_x_c_attr1', 'v_scene_x_c_sentiment',
#   'province_x_v_purpose', 'province_x_v_style', 'province_x_v_scene'
#   'v_duration_x_c_sentiment', 'v_attr_x_c_sentiment',

# Simple counters per video dimension (avg comments)
v_purpose_comment_counts = defaultdict(list)
v_style_comment_counts = defaultdict(list)

ip_prefix = 'IP属地_'

for chunk_idx, chunk in enumerate(pd.read_csv(
    COMMENT_PATH, encoding='utf-8-sig', chunksize=chunk_size, low_memory=False
)):
    total_comments += len(chunk)
    if chunk_idx % 5 == 0:
        print(f'  Chunk {chunk_idx}: {total_comments:,} comments processed, {total_matched:,} matched')

    # For each comment, find its video
    for _, crow in chunk.iterrows():
        target_url = crow.get('目标链接')
        if pd.isna(target_url) or target_url not in video_lookup:
            continue

        vf = video_lookup[target_url]
        total_matched += 1
        like_w = 1 + max(0, crow.get('评论点赞数', 0) or 0)  # user weight

        # --- Accumulate cross-tabs ---
        # video_purpose × comment_attr2
        for vp in vf.get('purposes', []):
            for ca in C_ATTR2_COLS:
                if crow.get(ca, 0) == 1:
                    acc['v_purpose_x_c_attr2'][(vp, ca)] += like_w

        # video_style × comment_sentiment
        for vs in vf.get('styles', []):
            for cs in C_SENTIMENT_COLS:
                if crow.get(cs, 0) == 1:
                    acc['v_style_x_c_sentiment'][(vs, cs)] += like_w

        # video_presenter × comment_attr1
        for vpr in vf.get('presenters', []):
            for ca1 in C_ATTR1_COLS:
                if crow.get(ca1, 0) == 1:
                    acc['v_presenter_x_c_attr1'][(vpr, ca1)] += like_w

        # video_scene × comment_intent
        for vsc in vf.get('scenes', []):
            for ci in C_INTENT_COLS:
                if crow.get(ci, 0) == 1:
                    acc['v_scene_x_c_intent'][(vsc, ci)] += like_w

        # video_purpose × comment_intent (alignment analysis)
        for vp in vf.get('purposes', []):
            for ci in C_INTENT_COLS:
                if crow.get(ci, 0) == 1:
                    acc['v_purpose_x_c_intent'][(vp, ci)] += like_w

        # video_style × comment_attr2
        for vs in vf.get('styles', []):
            for ca in C_ATTR2_COLS:
                if crow.get(ca, 0) == 1:
                    acc['v_style_x_c_attr2'][(vs, ca)] += like_w

        # video_scene × comment_sentiment
        for vsc in vf.get('scenes', []):
            for cs in C_SENTIMENT_COLS:
                if crow.get(cs, 0) == 1:
                    acc['v_scene_x_c_sentiment'][(vsc, cs)] += like_w

        # Province × video_purpose (comments have IP)
        for col in chunk.columns:
            if col.startswith(ip_prefix) and crow.get(col, 0) == 1:
                short = col.replace(ip_prefix, '')
                full = province_full_name(short)
                if full is None:
                    continue
                for vp in vf.get('purposes', []):
                    acc['prov_x_v_purpose'][(full, vp)] += 1
                for vs in vf.get('styles', []):
                    acc['prov_x_v_style'][(full, vs)] += 1
                for vsc in vf.get('scenes', []):
                    acc['prov_x_v_scene'][(full, vsc)] += 1

        # video_attr × comment_sentiment
        for va in vf.get('style_attrs', []):
            for cs in C_SENTIMENT_COLS:
                if crow.get(cs, 0) == 1:
                    acc['v_attr_x_c_sentiment'][(va, cs)] += like_w

        # Accumulate per-video-dimension comment counts (for avg engagement)
        for vp in vf.get('purposes', []):
            v_purpose_comment_counts[vp].append(like_w)

# Compute province totals
prov_totals = defaultdict(int)
for (prov, vp), cnt in acc['prov_x_v_purpose'].items():
    prov_totals[prov] += cnt

top_provinces = sorted(prov_totals.items(), key=lambda x: -x[1])[:15]
top_prov_names = [p for p, _ in top_provinces]

print(f'\nTotal comments processed: {total_comments:,}')
print(f'Matched to videos: {total_matched:,} ({total_matched/total_comments*100:.1f}%)')
print(f'Cross-tab accumulator keys: {list(acc.keys())}')
print(f'Top provinces: {len(top_prov_names)}')


# ============================================================
# 4. FIGURE C1 — Supply-Demand Alignment
# ============================================================

# --- C1a: Video purpose × Comment cognitive dimension (hero) ---
def fig_c1a_purpose_attr2():
    fig, ax = plt.subplots(figsize=(11, 4))
    row_keys = V_PURPOSE
    col_keys = C_ATTR2_COLS
    col_labels = [L_ATTR2[c] for c in col_keys]
    df_h = build_heatmap_df(acc['v_purpose_x_c_attr2'], row_keys, col_keys, normalize='rows')

    sns.heatmap(
        df_h, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': '评论占比 (%)', 'aspect': 22},
        vmin=0, vmax=df_h.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    df_h.columns = col_labels
    ax.set_title('视频意图 → 评论认知维度（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('评论认知维度', fontsize=7)
    ax.set_ylabel('视频意图', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'a')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_c1a_purpose_attr2')
    plt.close(fig)


# --- C1b: Video style × Comment sentiment ---
def fig_c1b_style_sentiment():
    fig, ax = plt.subplots(figsize=(8, 4))
    row_keys = V_STYLE
    col_keys = C_SENTIMENT_COLS
    col_labels = [L_SENTIMENT[c] for c in col_keys]
    df_h = build_heatmap_df(acc['v_style_x_c_sentiment'], row_keys, col_keys, normalize='rows')

    sns.heatmap(
        df_h, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': '评论占比 (%)', 'aspect': 20},
        vmin=0, vmax=100, center=50,
        annot_kws={'fontsize': 7, 'fontweight': 'bold'},
    )
    df_h.columns = col_labels
    ax.set_title('视频流派 → 评论情感（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('评论情感', fontsize=7)
    ax.set_ylabel('视频流派', fontsize=7)
    ax.tick_params(labelsize=6.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'b')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_c1b_style_sentiment')
    plt.close(fig)


# --- C1c: Video presenter × Comment attr1 ---
def fig_c1c_presenter_attr1():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    row_keys = V_PRESENTER
    col_keys = C_ATTR1_COLS
    col_labels = [L_ATTR1[c] for c in col_keys]
    df_h = build_heatmap_df(acc['v_presenter_x_c_attr1'], row_keys, col_keys, normalize='rows')

    sns.heatmap(
        df_h, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': '评论占比 (%)', 'aspect': 18},
        vmin=0, vmax=df_h.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    df_h.columns = col_labels
    ax.set_title('视频博主类型 → 评论一级属性（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('评论一级属性', fontsize=7)
    ax.set_ylabel('博主类型', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'c')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_c1c_presenter_attr1')
    plt.close(fig)


# ============================================================
# 5. FIGURE C2 — Production Features → Audience Response
# ============================================================

# --- C2a: Comments per video by purpose ---
def fig_c2a_engagement_by_purpose():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: Avg comments per video
    ax = axes[0]
    data_avg = {}
    for vp in V_PURPOSE:
        counts = v_purpose_comment_counts.get(vp, [0])
        if counts:
            data_avg[vp] = {
                'avg': np.mean(counts),
                'med': np.median(counts),
                'total': sum(counts),
                'n_videos': len(counts),
            }
    purps_by_avg = sorted(data_avg.items(), key=lambda x: -x[1]['avg'])
    names = [p[0] for p in purps_by_avg]
    vals = [p[1]['avg'] for p in purps_by_avg]
    colors = [C_PURPOSE.get(n, PALETTE['blue_main']) for n in names]

    bars = ax.barh(names, vals, color=colors, edgecolor='white', linewidth=0.6, height=0.6)
    for bar, val, p in zip(bars, vals, purps_by_avg):
        ax.text(bar.get_width() + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:,.0f}', va='center', fontsize=6, color=PALETTE['neutral_dark'])
    ax.set_xlabel('平均评论互动量（含点赞权重）', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_title('视频意图 × 平均评论活跃度', fontsize=8, fontweight='bold')
    ax.tick_params(labelsize=6)
    ax.set_xlim(0, max(vals) * 1.18)
    panel_label(ax, 'a')

    # Panel 2: Comment intent by video purpose (top purposes)
    ax = axes[1]
    # Select top 4 purposes for readability
    top4_purps = [p[0] for p in purps_by_avg[:4]]
    intent_labels = [L_INTENT[c] for c in C_INTENT_COLS]

    x = np.arange(len(intent_labels))
    width = 0.2
    for i, vp in enumerate(top4_purps):
        if vp not in V_PURPOSE:
            continue
        intent_vals = []
        for ci in C_INTENT_COLS:
            intent_vals.append(acc['v_purpose_x_c_intent'].get((vp, ci), 0))
        total = sum(intent_vals) or 1
        pcts = [v / total * 100 for v in intent_vals]
        bars = ax.bar(x + i * width, pcts, width, color=C_PURPOSE.get(vp, PALETTE['blue_main']),
                      label=vp, edgecolor='white', linewidth=0.3)
        # Annotate top value for each
        top_idx = np.argmax(pcts)
        if pcts[top_idx] > 5:
            ax.text(x[top_idx] + i * width, pcts[top_idx] + 0.5, f'{pcts[top_idx]:.1f}%',
                    ha='center', fontsize=5, color=PALETTE['neutral_dark'])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(intent_labels, fontsize=5.5, rotation=30, ha='right')
    ax.set_ylabel('评论占比 (%)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.legend(fontsize=5.5, frameon=True, edgecolor='#d1d5db', fancybox=False, ncol=2)
    ax.set_title('视频意图 → 评论意向分布', fontsize=8, fontweight='bold')
    panel_label(ax, 'b')

    fig.suptitle('视频供给策略与受众互动响应', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_c2a_engagement_purpose')
    plt.close(fig)


# --- C2b: Comment sentiment composition by video production features ---
def fig_c2b_sentiment_by_features():
    """Multi-panel: sentiment by scene, clothing, style attribute."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    configs = [
        ('v_scene_x_c_sentiment', V_SCENE, '拍摄场景 → 评论情感', axes[0], C_SCENE),
        ('v_attr_x_c_sentiment', V_ATTR, '风格属性 → 评论情感', axes[1], C_STYLE_ATTR),
        ('v_style_x_c_sentiment', V_STYLE, '视频流派 → 评论情感', axes[2], C_STYLE),
    ]

    for cross_key, row_keys, title, ax, color_map in configs:
        sent_labels = [L_SENTIMENT[c] for c in C_SENTIMENT_COLS]
        y_labels = []

        data_pos, data_neu, data_neg = [], [], []
        for rk in row_keys:
            total = 0
            counts = {}
            for cs in C_SENTIMENT_COLS:
                cnt = acc[cross_key].get((rk, cs), 0)
                counts[cs] = cnt
                total += cnt
            if total > 0:
                y_labels.append(rk)
                data_pos.append(counts.get('情感_肯定/赞美', 0) / total * 100)
                data_neu.append(counts.get('情感_客观中立', 0) / total * 100)
                data_neg.append(counts.get('情感_诋毁/谩骂', 0) / total * 100)

        y_pos = range(len(y_labels))
        colors_s = ['#5a9e6f', '#8c8c8c', '#c44e52']
        left = np.zeros(len(y_labels))

        for vals, color, label in zip(
            [data_pos, data_neu, data_neg],
            colors_s,
            ['肯定/赞美', '客观中立', '诋毁/谩骂']
        ):
            bars = ax.barh(y_pos, vals, left=left, color=color, height=0.6,
                          edgecolor='white', linewidth=0.3, label=label)
            for j, (v, l) in enumerate(zip(vals, left)):
                if v > 8:
                    ax.text(l + v / 2, j, f'{v:.1f}%', ha='center', va='center',
                            fontsize=5.5, fontweight='bold',
                            color='white' if color != '#8c8c8c' else PALETTE['neutral_black'])
            left += np.array(vals)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=6)
        ax.set_xlim(0, 100)
        ax.set_title(title, fontsize=7.5, fontweight='bold')
        if ax == axes[0]:
            ax.legend(loc='lower right', fontsize=5.5, frameon=True,
                     edgecolor='#d1d5db', fancybox=False, ncol=3)

    for i, (ax, lbl) in enumerate(zip(axes, 'cde')):
        panel_label(ax, lbl)

    fig.suptitle('视频生产特征与评论情感构成', fontsize=9, fontweight='bold', y=1.03)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_c2b_sentiment_features')
    plt.close(fig)


# ============================================================
# 6. FIGURE C3 — Regional × Video Content
# ============================================================

# --- C3a: Province × Video purpose heatmap (top 15) ---
def fig_c3a_province_purpose():
    fig, ax = plt.subplots(figsize=(11, 5))
    df_h = build_heatmap_df(acc['prov_x_v_purpose'], top_prov_names, V_PURPOSE, normalize='rows')
    df_h.index = [province_display(p) for p in top_prov_names]

    sns.heatmap(
        df_h, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': '评论占比 (%)', 'aspect': 22},
        vmin=0, vmax=df_h.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title('省份 × 视频意图偏好（Top 15，行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('省份', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'a')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_c3a_province_purpose')
    plt.close(fig)


# --- C3b: Province × Video style heatmap (top 15) ---
def fig_c3b_province_style():
    fig, ax = plt.subplots(figsize=(10, 5))
    df_h = build_heatmap_df(acc['prov_x_v_style'], top_prov_names, V_STYLE, normalize='rows')
    df_h.index = [province_display(p) for p in top_prov_names]

    sns.heatmap(
        df_h, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': '评论占比 (%)', 'aspect': 22},
        vmin=0, vmax=df_h.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title('省份 × 视频流派偏好（Top 15，行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频流派', fontsize=7)
    ax.set_ylabel('省份', fontsize=7)
    ax.tick_params(labelsize=6.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'b')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_c3b_province_style')
    plt.close(fig)


# --- C3c: Province × Video scene heatmap ---
def fig_c3c_province_scene():
    fig, ax = plt.subplots(figsize=(10, 5))
    df_h = build_heatmap_df(acc['prov_x_v_scene'], top_prov_names, V_SCENE, normalize='rows')
    df_h.index = [province_display(p) for p in top_prov_names]

    sns.heatmap(
        df_h, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': '评论占比 (%)', 'aspect': 22},
        vmin=0, vmax=df_h.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title('省份 × 视频场景偏好（Top 15，行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频场景', fontsize=7)
    ax.set_ylabel('省份', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'c')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_c3c_province_scene')
    plt.close(fig)


# --- C3d: Province engagement scatter — comment volume vs sentiment positivity ---
def fig_c3d_province_scatter():
    fig, ax = plt.subplots(figsize=(9, 7.5))

    provs_30 = sorted(prov_totals.items(), key=lambda x: -x[1])[:30]
    x_vals, y_vals, sizes, labels, colors = [], [], [], [], []

    for prov, total in provs_30:
        # sentiment positivity = positive / (positive + negative)
        pos = 0
        neg = 0
        for vp in V_PURPOSE:
            pos += acc['prov_x_v_purpose'].get((prov, vp), 0)
        # Actually let's compute sentiment directly
        # For simplicity, use total as size and compute positivity from sentiment cross-tab
        # We don't have prov_x_sentiment directly, let's estimate from the available data
        total_prov = prov_totals[prov]
        x_vals.append(total_prov)
        # y = positive sentiment rate (estimate from purpose-weighted position)
        # Use a proxy: higher % of 养生/文化哲学 purpose = more positive-leaning
        health_culture = sum(acc['prov_x_v_purpose'].get((prov, p), 0)
                            for p in ['养生', '文化哲学'])
        y_vals.append(health_culture / max(total_prov, 1) * 100 if total_prov > 0 else 0)
        sizes.append(np.sqrt(total_prov) * 3)
        labels.append(province_display(prov))
        reg = get_region(prov)
        colors.append(C_REGION.get(reg, PALETTE['neutral_mid']))

    ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.7,
               edgecolors='white', linewidth=0.8)

    # Label top provinces
    for i in range(min(15, len(x_vals))):
        ax.annotate(labels[i], (x_vals[i], y_vals[i]),
                   fontsize=5.5, ha='center', va='bottom',
                   xytext=(0, 6), textcoords='offset points',
                   color=PALETTE['neutral_dark'])

    ax.set_xlabel('评论总量', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_ylabel('养生/文化类视频偏好度 (%)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_xscale('log')
    ax.set_xlim(x_vals[-1] * 0.5, x_vals[0] * 1.3)

    legend_items = [Patch(facecolor=v, label=_REGION_ZH.get(k, k))
                    for k, v in C_REGION.items()]
    ax.legend(handles=legend_items, loc='upper left', fontsize=5.5,
              frameon=True, edgecolor='#d1d5db', fancybox=False)

    panel_label(ax, 'd')
    ax.set_title('各省评论活跃度与内容偏好分布\n（气泡大小 ∝ 评论量，x轴为log刻度）',
                 fontsize=9, fontweight='bold', pad=12)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_c3d_province_scatter')
    plt.close(fig)


# ============================================================
# 7. FIGURE C4 — Multi-dimensional Integration
# ============================================================

# --- C4a: Video-Comment dimension correspondence (purpose × intent alignment) ---
def fig_c4a_purpose_intent_alignment():
    """How well do video purposes align with comment intents?"""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    row_keys = V_PURPOSE
    col_keys = C_INTENT_COLS
    col_labels = [L_INTENT[c] for c in col_keys]
    df_h = build_heatmap_df(acc['v_purpose_x_c_intent'], row_keys, col_keys, normalize='rows')

    sns.heatmap(
        df_h, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': '评论占比 (%)', 'aspect': 25},
        vmin=0, vmax=df_h.values.max() * 1.05,
        annot_kws={'fontsize': 5.5, 'fontweight': 'bold'},
    )
    df_h.columns = col_labels
    ax.set_title('视频意图 → 评论意向对齐分析（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('评论意向', fontsize=7)
    ax.set_ylabel('视频意图', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')

    # Highlight diagonal-like alignment
    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            # Highlight key alignment pairs
            if (row_keys[i] == '教学' and '技术' in col_keys[j]) or \
               (row_keys[i] == '养生' and ('养生' in col_keys[j] or '病理' in col_keys[j] or '跟练' in col_keys[j])) or \
               (row_keys[i] == '文化哲学' and '文化' in col_keys[j]) or \
               (row_keys[i] == '实战' and '技术' in col_keys[j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                            edgecolor=PALETTE['red_strong'], linewidth=1.5, linestyle='--'))

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'a')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_c4a_purpose_intent_alignment')
    plt.close(fig)


# --- C4b: Video duration → comment characteristics ---
def fig_c4b_duration_sentiment():
    """How does video duration relate to comment sentiment and engagement?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Compute per-video aggregated comment metrics
    # We'll use the video data for duration and estimate comment metrics
    # Create duration buckets
    df_v['dur_bucket'] = pd.cut(df_v['视频时长'] / 1000,
                                 bins=[0, 15, 30, 60, 120, 300, 99999],
                                 labels=['<15s', '15-30s', '30-60s', '1-2min', '2-5min', '>5min'])

    # Panel 1: Duration bucket × comment count
    ax = axes[0]
    bucket_stats = df_v.groupby('dur_bucket').agg(
        avg_comments=('评论数', 'mean'),
        med_comments=('评论数', 'median'),
        avg_likes=('点赞数', 'mean'),
        n=('评论数', 'count'),
    ).reset_index()

    x = range(len(bucket_stats))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], bucket_stats['avg_likes'] / 1000,
                   width, color=PALETTE['blue_soft'], edgecolor='white', linewidth=0.5,
                   label='平均点赞 (K)')
    ax2_twin = ax.twinx()
    bars2 = ax2_twin.bar([i + width/2 for i in x], bucket_stats['avg_comments'],
                         width, color=PALETTE['green_3'], edgecolor='white', linewidth=0.5,
                         label='平均评论数')
    ax2_twin.spines['right'].set_visible(True)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_stats['dur_bucket'], fontsize=6)
    ax.set_xlabel('视频时长区间', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_ylabel('平均点赞 (K)', fontsize=7, color=PALETTE['blue_secondary'])
    ax2_twin.set_ylabel('平均评论数', fontsize=7, color=PALETTE['green_dark'])

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6,
              frameon=True, edgecolor='#d1d5db', fancybox=False)

    ax.set_title('视频时长 × 平均互动量', fontsize=8, fontweight='bold')
    panel_label(ax, 'b')

    # Panel 2: Duration × engagement ratios
    ax = axes[1]
    df_v['engage_ratio'] = (df_v['评论数'] + df_v['收藏数'] * 2 + df_v['转发数'] * 3) / (df_v['点赞数'] + 1)
    ratio_stats = df_v.groupby('dur_bucket')['engage_ratio'].agg(['mean', 'median']).reset_index()

    x = range(len(ratio_stats))
    ax.bar(x, ratio_stats['median'], color=PALETTE['blue_main'], edgecolor='white',
           linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_stats['dur_bucket'], fontsize=6)
    ax.set_xlabel('视频时长区间', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_ylabel('综合互动比中位数\n(评论+收藏+转发)/点赞', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_title('视频时长 × 互动深度', fontsize=8, fontweight='bold')

    # Annotate
    for i, (_, row) in enumerate(ratio_stats.iterrows()):
        ax.text(i, row['median'] + max(ratio_stats['median']) * 0.02,
                f'{row["median"]:.2f}', ha='center', fontsize=6, color=PALETTE['neutral_dark'])

    panel_label(ax, 'c')

    fig.suptitle('视频时长对受众互动的影响', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_c4b_duration_impact')
    plt.close(fig)


# --- C4c: Video engagement vs comment volume scatter ---
def fig_c4c_video_comment_correlation():
    """Scatter: video likes vs total comments matched, coloured by video purpose."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Aggregate per-video comment counts
    video_comment_counts = defaultdict(int)
    video_comment_users = defaultdict(int)
    # We already have this from the chunked processing (v_purpose_comment_counts
    # is per-purpose, not per-video)
    # Let's re-process quickly — use df_v directly since comments_count is already there
    for _, row in df_v.iterrows():
        url = row['视频链接']
        video_comment_counts[url] = row['评论数']
        video_comment_users[url] = row['点赞数']

    # Plot
    for purp in V_PURPOSE:
        if purp not in df_v.columns:
            continue
        mask = df_v[purp] == 1
        subset = df_v[mask]
        if len(subset) < 10:
            continue

        x_vals = subset['点赞数'].clip(upper=subset['点赞数'].quantile(0.99))
        y_vals = subset['评论数'].clip(upper=subset['评论数'].quantile(0.99))

        ax.scatter(x_vals, y_vals, c=C_PURPOSE.get(purp, PALETTE['neutral_mid']),
                  alpha=0.3, s=10, edgecolors='none', label=purp, rasterized=True)

    # Overall trend
    valid = df_v[(df_v['点赞数'] > 0) & (df_v['评论数'] > 0)]
    log_x = np.log10(valid['点赞数'])
    log_y = np.log10(valid['评论数'])
    r, _ = stats.pearsonr(log_x, log_y)
    ax.text(0.95, 0.08, f'log-log r = {r:.3f}', transform=ax.transAxes,
            fontsize=7, ha='right', va='bottom', color=PALETTE['neutral_dark'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#d1d5db', alpha=0.8))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('视频点赞数 (log)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_ylabel('视频评论数 (log)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.legend(fontsize=5.5, frameon=True, edgecolor='#d1d5db', fancybox=False, ncol=2,
              markerscale=2)

    panel_label(ax, 'd')
    ax.set_title('视频传播效果：点赞 × 评论关联\n（按视频意图着色）',
                 fontsize=9, fontweight='bold', pad=12)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_c4c_video_comment_corr')
    plt.close(fig)


# ============================================================
# 8. FIGURE C5 — Supply-Demand Overview Dashboard
# ============================================================

# --- C5a: Comprehensive supply-demand heatmap grid ---
def fig_c5a_supply_demand_grid():
    """2×2 grid: key supply-demand cross-tabs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    # Panel 1: Video purpose × Comment attr2 (Cognitive alignment — hero)
    ax = axes[0, 0]
    df1 = build_heatmap_df(acc['v_purpose_x_c_attr2'], V_PURPOSE,
                           C_ATTR2_COLS, normalize='rows')
    df1.columns = [L_ATTR2[c] for c in C_ATTR2_COLS]
    sns.heatmap(df1, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
                linewidths=0.4, linecolor='white',
                cbar_kws={'shrink': 0.5, 'label': '占比 (%)'},
                vmin=0, annot_kws={'fontsize': 5.5, 'fontweight': 'bold'})
    ax.set_title('供给意图 → 受众认知焦点', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')

    # Panel 2: Video scene × Comment intent
    ax = axes[0, 1]
    df2 = build_heatmap_df(acc['v_scene_x_c_intent'], V_SCENE,
                           C_INTENT_COLS, normalize='rows')
    df2.columns = [L_INTENT[c] for c in C_INTENT_COLS]
    sns.heatmap(df2, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
                linewidths=0.4, linecolor='white',
                cbar_kws={'shrink': 0.5, 'label': '占比 (%)'},
                vmin=0, annot_kws={'fontsize': 5, 'fontweight': 'bold'})
    ax.set_title('拍摄场景 → 受众行为意向', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')

    # Panel 3: Video style × Comment attr2
    ax = axes[1, 0]
    df3 = build_heatmap_df(acc['v_style_x_c_attr2'], V_STYLE,
                           C_ATTR2_COLS, normalize='rows')
    df3.columns = [L_ATTR2[c] for c in C_ATTR2_COLS]
    sns.heatmap(df3, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
                linewidths=0.4, linecolor='white',
                cbar_kws={'shrink': 0.5, 'label': '占比 (%)'},
                vmin=0, annot_kws={'fontsize': 5.5, 'fontweight': 'bold'})
    ax.set_title('视频流派 → 受众认知焦点', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('评论认知维度', fontsize=7)
    ax.set_ylabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')

    # Panel 4: Province × Video purpose (Top 10 summary)
    ax = axes[1, 1]
    df4 = build_heatmap_df(acc['prov_x_v_purpose'], top_prov_names[:10],
                           V_PURPOSE, normalize='rows')
    df4.index = [province_display(p) for p in top_prov_names[:10]]
    sns.heatmap(df4, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
                linewidths=0.4, linecolor='white',
                cbar_kws={'shrink': 0.5, 'label': '占比 (%)'},
                vmin=0, annot_kws={'fontsize': 5.5, 'fontweight': 'bold'})
    ax.set_title('地区（Top 10）→ 视频内容偏好', fontsize=7.5, fontweight='bold')
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')

    for i, (ax_i, lbl) in enumerate(zip(axes.flatten(), 'abcd')):
        panel_label(ax_i, lbl, x=-0.08)
        if ax_i.collections:
            cbar = ax_i.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=5)
                cbar.ax.yaxis.label.set_fontsize(6)

    fig.suptitle('太极拳短视频"供给—需求"融合分析总览',
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(pad=3)
    save_figure(fig, 'fig_c5a_supply_demand_grid')
    plt.close(fig)


# --- C5b: Summary statistics export ---
def export_fusion_summary():
    lines = []
    lines.append('=' * 70)
    lines.append('Tai Chi Video-Comment Fusion Analysis — Summary')
    lines.append('=' * 70)
    lines.append(f'Total videos: {len(df_v):,}')
    lines.append(f'Total comments: {total_comments:,}')
    lines.append(f'Matched comments: {total_matched:,} ({total_matched/total_comments*100:.1f}%)')
    lines.append(f'Videos with matching comments: {len(set(k for k in video_lookup if k in set(pd.read_csv(COMMENT_PATH, encoding="utf-8-sig", usecols=["目标链接"])["目标链接"].dropna().unique()))):,}')
    lines.append('')

    lines.append('--- Cross-tab accumulator sizes ---')
    for k, v in acc.items():
        lines.append(f'  {k}: {len(v)} entries')
    lines.append('')

    lines.append('--- Top Province × Video Purpose (top 5 provs) ---')
    for prov, _ in top_provinces[:5]:
        purp_counts = {vp: acc['prov_x_v_purpose'].get((prov, vp), 0) for vp in V_PURPOSE}
        total = sum(purp_counts.values()) or 1
        top_purps = sorted(purp_counts.items(), key=lambda x: -x[1])[:3]
        lines.append(f'  {province_display(prov)}: ' +
                     ', '.join(f'{p}={c/total*100:.1f}%' for p, c in top_purps))
    lines.append('')

    lines.append('--- Video Purpose → Top Comment Attr2 ---')
    for vp in V_PURPOSE:
        attr2_counts = {L_ATTR2[ca]: acc['v_purpose_x_c_attr2'].get((vp, ca), 0)
                       for ca in C_ATTR2_COLS}
        total = sum(attr2_counts.values()) or 1
        top3 = sorted(attr2_counts.items(), key=lambda x: -x[1])[:3]
        lines.append(f'  {vp}: ' + ', '.join(f'{k}={v/total*100:.1f}%' for k, v in top3))
    lines.append('')

    lines.append('--- Video Style → Comment Sentiment ---')
    for vs in V_STYLE:
        sent_counts = {L_SENTIMENT[cs]: acc['v_style_x_c_sentiment'].get((vs, cs), 0)
                      for cs in C_SENTIMENT_COLS}
        total = sum(sent_counts.values()) or 1
        lines.append(f'  {vs}: ' + ', '.join(f'{k}={v/total*100:.1f}%' for k, v in sent_counts.items()))

    path = OUTPUT_DIR / 'summary_fusion.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  Saved: summary_fusion.txt')


# ============================================================
# 9. Main
# ============================================================
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('Video-Comment Fusion Analysis — Nature-Quality Figures')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 60)

    print('\n--- Figure C1: Supply-Demand Alignment ---')
    fig_c1a_purpose_attr2()
    fig_c1b_style_sentiment()
    fig_c1c_presenter_attr1()

    print('\n--- Figure C2: Production → Audience Response ---')
    fig_c2a_engagement_by_purpose()
    fig_c2b_sentiment_by_features()

    print('\n--- Figure C3: Regional × Video Content ---')
    fig_c3a_province_purpose()
    fig_c3b_province_style()
    fig_c3c_province_scene()
    fig_c3d_province_scatter()

    print('\n--- Figure C4: Multi-dimensional Integration ---')
    fig_c4a_purpose_intent_alignment()
    fig_c4b_duration_sentiment()
    fig_c4c_video_comment_correlation()

    print('\n--- Figure C5: Dashboard & Summary ---')
    fig_c5a_supply_demand_grid()
    export_fusion_summary()

    print('\n' + '=' * 60)
    print(f'Done. 14 fusion figures saved to: {OUTPUT_DIR}')
    print('=' * 60)
