"""
=============================================================================
Nature-quality figures: Tai Chi Cultural Heritage Cognitive Analysis
Arguments 2 & 3 — Cognition, value identification, geographic distribution
=============================================================================
Input: merged_dy_data_Allcomments_labeled_cleaned_onehot.csv (1.47M rows × 73 cols)
Output: dy_file/dy_video_csv/figures/ — SVG + PDF + TIFF per figure
Backend: Python (matplotlib + seaborn)
=============================================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import COMMENT_LABELED_CLEANED_ONEHOT, FIGURES_COMMENT_DIR

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
INPUT_PATH = str(COMMENT_LABELED_CLEANED_ONEHOT)
OUTPUT_DIR = FIGURES_COMMENT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANG = 'zh'  # 'zh' = Chinese labels, 'en' = English labels

# ----------------------------------------------------------
# Nature-style rcParams
# ----------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",       # editable text in SVG
    "pdf.fonttype": 42,           # editable TrueType in PDF
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
# 1. Colour palettes (Nature low-saturation unified families)
# ============================================================
C_SENTIMENT = {'肯定/赞美': '#5a9e6f', '诋毁/谩骂': '#c44e52', '客观中立': '#8c8c8c'}
C_SENTIMENT_EN = {'Affirmation': '#5a9e6f', 'Criticism': '#c44e52', 'Neutral': '#8c8c8c'}

C_BAR_MAIN = '#5d8aa8'        # unified bar colour
C_BAR_HERO = '#3a6b8c'        # hero bar accent

C_REGION = {
    'North China':       '#c44e52',
    'Northeast':         '#dd8452',
    'East China':        '#55a868',
    'Central China':     '#4c72b0',
    'South China':       '#937860',
    'Southwest':         '#64b5cd',
    'Northwest':         '#da8bc3',
    'HK/Macau/Taiwan':   '#8c8c8c',
}

C_HEATMAP = sns.color_palette("YlGnBu", 256)  # multi-hue: yellow → green → blue

# ============================================================
# 2. Column definitions
# ============================================================
SENTIMENT_COLS = ['情感_肯定/赞美', '情感_诋毁/谩骂', '情感_客观中立']
ATTR1_COLS_MAIN = [
    '属性一_尊师重道', '属性一_实战性质疑', '属性一_文化质疑', '属性一_文化捍卫',
    '属性一_玩网络梗', '属性一_阴阳怪气', '属性一_对博主人身攻击', '属性一_对博主性骚扰',
]
ATTR2_COLS_MAIN = [
    '属性二_动作要点', '属性二_实战技术', '属性二_养生功效', '属性二_视觉审美',
    '属性二_文化哲学', '属性二_博主本身', '属性二_关注其他人物', '属性二_关注配乐审美',
    '属性二_关注跨拳种实战比较',
]
INTENT_COLS_MAIN = [
    '意向_种草意向', '意向_资源索取', '意向_技术追问', '意向_文化探讨',
    '意向_跟练体感', '意向_病理/生理求助', '意向_寻师求学', '意向_主动推荐',
    '意向_影视文学跨界探讨', '意向_门派流派探讨',
]

ALL_SENTIMENT = SENTIMENT_COLS
ALL_ATTR1 = ATTR1_COLS_MAIN + ['属性一_其他']
ALL_ATTR2 = ATTR2_COLS_MAIN + ['属性二_其他']
ALL_INTENT = INTENT_COLS_MAIN + ['意向_其他']

# ============================================================
# 3. Bilingual label maps
# ============================================================
_ZH = {
    '情感_肯定/赞美': '肯定/赞美', '情感_诋毁/谩骂': '诋毁/谩骂', '情感_客观中立': '客观中立',
    '属性一_尊师重道': '尊师重道', '属性一_实战性质疑': '实战性质疑',
    '属性一_文化质疑': '文化质疑', '属性一_文化捍卫': '文化捍卫',
    '属性一_玩网络梗': '网络玩梗', '属性一_阴阳怪气': '阴阳怪气',
    '属性一_对博主人身攻击': '人身攻击', '属性一_对博主性骚扰': '性骚扰',
    '属性二_动作要点': '动作要点', '属性二_实战技术': '实战技术',
    '属性二_养生功效': '养生功效', '属性二_视觉审美': '视觉审美',
    '属性二_文化哲学': '文化哲学', '属性二_博主本身': '博主本身',
    '属性二_关注其他人物': '关注其他人物', '属性二_关注配乐审美': '配乐审美',
    '属性二_关注跨拳种实战比较': '跨拳种比较',
    '意向_种草意向': '种草意向', '意向_资源索取': '资源索取',
    '意向_技术追问': '技术追问', '意向_文化探讨': '文化探讨',
    '意向_跟练体感': '跟练体感', '意向_病理/生理求助': '病理/生理求助',
    '意向_寻师求学': '寻师求学', '意向_主动推荐': '主动推荐',
    '意向_影视文学跨界探讨': '影视文学探讨', '意向_门派流派探讨': '门派流派探讨',
}
_EN = {
    '情感_肯定/赞美': 'Affirmation', '情感_诋毁/谩骂': 'Criticism', '情感_客观中立': 'Neutral',
    '属性一_尊师重道': 'Respect Teachers', '属性一_实战性质疑': 'Combat Skepticism',
    '属性一_文化质疑': 'Cultural Skepticism', '属性一_文化捍卫': 'Cultural Defense',
    '属性一_玩网络梗': 'Internet Memes', '属性一_阴阳怪气': 'Sarcasm',
    '属性一_对博主人身攻击': 'Personal Attack', '属性一_对博主性骚扰': 'Harassment',
    '属性二_动作要点': 'Movement Tech.', '属性二_实战技术': 'Combat Skills',
    '属性二_养生功效': 'Health Benefits', '属性二_视觉审美': 'Visual Aesthetics',
    '属性二_文化哲学': 'Cultural Philosophy', '属性二_博主本身': 'Content Creator',
    '属性二_关注其他人物': 'Other Figures', '属性二_关注配乐审美': 'Music Aesthetics',
    '属性二_关注跨拳种实战比较': 'Cross-style Comp.',
    '意向_种草意向': 'Interest', '意向_资源索取': 'Resource Request',
    '意向_技术追问': 'Technical Inquiry', '意向_文化探讨': 'Cultural Discussion',
    '意向_跟练体感': 'Practice Exp.', '意向_病理/生理求助': 'Health Advice',
    '意向_寻师求学': 'Seek Teachers', '意向_主动推荐': 'Recommendation',
    '意向_影视文学跨界探讨': 'Film/Literature', '意向_门派流派探讨': 'School/Style',
}


def T(zh_text, en_text):
    return zh_text if LANG == 'zh' else en_text


def L(key):
    return _ZH[key] if LANG == 'zh' else _EN[key]


# ============================================================
# 4. Province & region mappings
# ============================================================
# Short IP name → Chinese full administrative name
PROVINCE_SHORT_MAP = {
    '北京': '北京市', '天津': '天津市', '上海': '上海市', '重庆': '重庆市',
    '广西': '广西壮族自治区', '新疆': '新疆维吾尔自治区',
    '宁夏': '宁夏回族自治区', '西藏': '西藏自治区', '内蒙古': '内蒙古自治区',
    '中国香港': '香港特别行政区', '中国澳门': '澳门特别行政区', '中国台湾': '台湾省',
}

PROVINCE_ORDER = [
    '北京市', '天津市', '河北省', '山西省', '内蒙古自治区',
    '辽宁省', '吉林省', '黑龙江省',
    '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省',
    '河南省', '湖北省', '湖南省',
    '广东省', '广西壮族自治区', '海南省',
    '重庆市', '四川省', '贵州省', '云南省', '西藏自治区',
    '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区',
    '香港特别行政区', '澳门特别行政区', '台湾省',
]

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


def region_label(en_key):
    return _REGION_ZH[en_key] if LANG == 'zh' else en_key


def province_full_name(short):
    """Map IP short name → ECharts full administrative name."""
    if short in PROVINCE_SHORT_MAP:
        return PROVINCE_SHORT_MAP[short]
    if short in ('中国', 'IP未知'):
        return None
    return short + '省'


def province_display(full_cn):
    """Strip administrative suffixes for compact display."""
    s = full_cn.replace('特别行政区', '').replace('自治区', '').replace('省', '').replace('市', '')
    display_map = {
        '中国香港': '香港', '中国澳门': '澳门', '中国台湾': '台湾',
        '内蒙古': '内蒙古', '广西': '广西', '新疆': '新疆', '宁夏': '宁夏', '西藏': '西藏',
    }
    return display_map.get(s, s)


# ============================================================
# 5. Data loading (chunked, single-pass aggregation)
# ============================================================
print('=' * 60)
print('Step 1: Loading data (chunked aggregation)...')
print('=' * 60)

chunk_size = 100000
aggregates = {}
total_rows = 0
total_likes = 0
province_aggregates = {}

# Cross-tab accumulators
sent_x_attr2 = {}
attr1_x_attr2 = {}
prov_x_attr2 = {}
prov_x_intent = {}
prov_x_sent = {}
sent_x_intent = {}
attr1_x_intent = {}

# Temporal & engagement accumulators (for exploratory figures)
monthly_total = {}
monthly_sent = {}
category_likes_raw = {}   # raw likes per category

all_count_cols = ALL_SENTIMENT + ALL_ATTR1 + ALL_ATTR2 + ALL_INTENT
for c in all_count_cols:
    aggregates[c] = 0

ip_prefix = 'IP属地_'

for chunk_idx, chunk in enumerate(pd.read_csv(
    INPUT_PATH, encoding='utf-8-sig', chunksize=chunk_size, low_memory=False
)):
    total_rows += len(chunk)
    likes = chunk['评论点赞数'].fillna(0).clip(upper=10000)
    chunk_likes = int(likes.sum())
    total_likes += chunk_likes
    if chunk_idx % 5 == 0:
        print(f'  Chunk {chunk_idx}, rows: {total_rows:,}, likes: {total_likes:,}')

    like_weight = 1 + likes

    # Aggregate category counts
    for c in all_count_cols:
        if c in chunk.columns:
            aggregates[c] += int((chunk[c] * like_weight).sum())

    # Province aggregates
    ip_cols = [c for c in chunk.columns if c.startswith(ip_prefix) and c != 'IP属地_IP未知']
    for ipc in ip_cols:
        short = ipc.replace(ip_prefix, '')
        full = province_full_name(short)
        if full is None:
            continue
        province_aggregates[full] = province_aggregates.get(full, 0) + int(chunk[ipc].sum())

    # Cross-tab: sentiment × attr2
    for sc in SENTIMENT_COLS:
        for ac in ALL_ATTR2:
            if sc in chunk.columns and ac in chunk.columns:
                mask = (chunk[sc] == 1) & (chunk[ac] == 1)
                key = (sc, ac)
                sent_x_attr2[key] = sent_x_attr2.get(key, 0) + int((mask * like_weight).sum())

    # Cross-tab: attr1 × attr2
    for a1c in ALL_ATTR1:
        for a2c in ALL_ATTR2:
            if a1c in chunk.columns and a2c in chunk.columns:
                mask = (chunk[a1c] == 1) & (chunk[a2c] == 1)
                key = (a1c, a2c)
                attr1_x_attr2[key] = attr1_x_attr2.get(key, 0) + int((mask * like_weight).sum())

    # Cross-tab: province × attr2, intent, sentiment
    for ipc in ip_cols:
        short = ipc.replace(ip_prefix, '')
        full = province_full_name(short)
        if full is None:
            continue
        for ac in ALL_ATTR2:
            if ac in chunk.columns:
                mask = (chunk[ipc] == 1) & (chunk[ac] == 1)
                key = (full, ac)
                prov_x_attr2[key] = prov_x_attr2.get(key, 0) + int(mask.sum())
        for ic in ALL_INTENT:
            if ic in chunk.columns:
                mask = (chunk[ipc] == 1) & (chunk[ic] == 1)
                key = (full, ic)
                prov_x_intent[key] = prov_x_intent.get(key, 0) + int(mask.sum())
        for sc in SENTIMENT_COLS:
            if sc in chunk.columns:
                mask = (chunk[ipc] == 1) & (chunk[sc] == 1)
                key = (full, sc)
                prov_x_sent[key] = prov_x_sent.get(key, 0) + int(mask.sum())

    # Cross-tab: sentiment × intent
    for sc in SENTIMENT_COLS:
        for ic in ALL_INTENT:
            if sc in chunk.columns and ic in chunk.columns:
                mask = (chunk[sc] == 1) & (chunk[ic] == 1)
                key = (sc, ic)
                sent_x_intent[key] = sent_x_intent.get(key, 0) + int((mask * like_weight).sum())

    # Cross-tab: attr1 × intent
    for a1c in ALL_ATTR1:
        for ic in ALL_INTENT:
            if a1c in chunk.columns and ic in chunk.columns:
                mask = (chunk[a1c] == 1) & (chunk[ic] == 1)
                key = (a1c, ic)
                attr1_x_intent[key] = attr1_x_intent.get(key, 0) + int((mask * like_weight).sum())

    # Temporal: monthly comment counts + monthly sentiment
    if '评论时间' in chunk.columns:
        months = pd.to_datetime(chunk['评论时间'], errors='coerce').dt.strftime('%Y-%m')
        for m, cnt in months.value_counts().items():
            if m != 'NaT' and pd.notna(m):
                monthly_total[m] = monthly_total.get(m, 0) + cnt
        for sc in SENTIMENT_COLS:
            if sc in chunk.columns:
                sent_months = months[chunk[sc] == 1].dropna()
                for m, cnt in sent_months.value_counts().items():
                    monthly_sent[(m, sc)] = monthly_sent.get((m, sc), 0) + cnt

    # Engagement: raw likes per category (for avg-likes analysis)
    raw_likes = chunk['评论点赞数'].fillna(0).clip(upper=10000)
    for c in ATTR2_COLS_MAIN + INTENT_COLS_MAIN:
        if c in chunk.columns:
            category_likes_raw[c] = category_likes_raw.get(c, 0) + int((chunk[c] * raw_likes).sum())


print(f'\nTotal comments (real users): {total_rows:,}')
print(f'Total likes: {total_likes:,}')
print(f'Total users (real + likes): {total_rows + total_likes:,}')
print(f'Provinces with data: {len(province_aggregates)}')
print(f'Comments with IP: {sum(province_aggregates.values()):,} '
      f'({sum(province_aggregates.values())/total_rows*100:.1f}%)')

# Province DataFrame
prov_df = pd.DataFrame({
    'province': list(province_aggregates.keys()),
    'total': list(province_aggregates.values()),
}).sort_values('total', ascending=False).reset_index(drop=True)

TOTAL_USERS = total_rows + total_likes

# Temporal summary
if monthly_total:
    months_sorted = sorted(monthly_total.keys())
    print(f'Time range: {months_sorted[0]} ~ {months_sorted[-1]} ({len(months_sorted)} months)')

# Engagement summary
total_raw_likes = int(sum(category_likes_raw.values()))
print(f'Total raw likes in categories: {total_raw_likes:,}')

# ============================================================
# 6. Export helper
# ============================================================
def save_figure(fig, filename_stem):
    """Save figure as SVG + PDF + high-DPI TIFF."""
    base = OUTPUT_DIR / filename_stem
    fig.savefig(str(base) + '.svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(str(base) + '.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(str(base) + '.tiff', dpi=600, bbox_inches='tight', facecolor='white',
                edgecolor='none', pil_kwargs={'compression': 'tiff_lzw'})
    print(f'  Saved: {filename_stem}{{.svg,.pdf,.tiff}}')


def build_series(cols, agg_dict):
    """Build a named Series from column keys, sorted ascending (for barh)."""
    data = {L(c): agg_dict[c] for c in cols}
    return pd.Series(data).sort_values(ascending=True)


def build_cross_table(col_list_a, col_list_b, cross_dict, normalize='rows'):
    """Build a cross-tabulation DataFrame, optionally row-normalised."""
    rows = [L(c) for c in col_list_a]
    cols = [L(c) for c in col_list_b]
    mat = np.zeros((len(rows), len(cols)))
    for i, ca in enumerate(col_list_a):
        for j, cb in enumerate(col_list_b):
            mat[i, j] = cross_dict.get((ca, cb), 0)
    df = pd.DataFrame(mat, index=rows, columns=cols)
    if normalize == 'rows':
        df = df.div(df.sum(axis=1), axis=0) * 100
    elif normalize == 'cols':
        df = df.div(df.sum(axis=0), axis=1) * 100
    return df


# ============================================================
# 7. FIGURE FUNCTIONS — Argument 2: Cognition & Value Identity
# ============================================================

# --- Fig 2a: Sentiment donut ---
def fig2a_sentiment_donut():
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    sent_s = build_series(SENTIMENT_COLS, aggregates)
    sent_keys = [L(c) for c in SENTIMENT_COLS]
    sent_vals = [aggregates[c] for c in SENTIMENT_COLS]
    colors = [C_SENTIMENT.get(k, '#8c8c8c') for k in sent_keys]

    wedges, texts = ax.pie(
        sent_vals, labels=None, startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.38, edgecolor='white', linewidth=2),
    )
    # Centre: total N
    ax.text(0, 0, f'{TOTAL_USERS:,}', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#2c3e50')
    ax.text(0, -0.18, T('用户总数', 'Total users'), ha='center', va='center',
            fontsize=6.5, color='#64748b')

    # Legend with counts & percentages
    pcts = [v / TOTAL_USERS * 100 for v in sent_vals]
    legend_labels = [
        f'{k}  ({v:,}, {p:.1f}%)'
        for k, v, p in zip(sent_keys, sent_vals, pcts)
    ]
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(0.88, 0.5),
              fontsize=6.5, frameon=False, handlelength=1.2, handleheight=1.2)

    ax.set_title(T('(a) 评论情感分布', '(a) Sentiment Distribution of Comments'),
                 fontsize=9, fontweight='bold', pad=14)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig2a_sentiment_donut')
    plt.close(fig)


# --- Fig 2b: Primary attributes bar ---
def fig2b_attr1_bar():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    s = build_series(ATTR1_COLS_MAIN, aggregates)
    colors = sns.color_palette("crest", n_colors=len(s))
    bars = ax.barh(s.index, s.values, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.62)
    max_v = s.values.max()
    for bar, val in zip(bars, s.values):
        pct = val / TOTAL_USERS * 100
        ax.text(bar.get_width() + max_v * 0.018, bar.get_y() + bar.get_height() / 2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=6, color='#334155')
    ax.set_xlim(0, max_v * 1.28)
    ax.set_xlabel(T('用户数（含点赞）', 'Users (incl. likes)'), fontsize=7, color='#475569')
    ax.set_title(T('(b) 一级属性分布', '(b) Distribution of Primary Attributes'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig2b_attr1_bar')
    plt.close(fig)


# --- Fig 2c: Cognitive dimensions bar ---
def fig2c_attr2_bar():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    s = build_series(ATTR2_COLS_MAIN, aggregates)
    colors = sns.color_palette("crest", n_colors=len(s))
    bars = ax.barh(s.index, s.values, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.62)
    max_v = s.values.max()
    for bar, val in zip(bars, s.values):
        pct = val / TOTAL_USERS * 100
        ax.text(bar.get_width() + max_v * 0.018, bar.get_y() + bar.get_height() / 2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=6, color='#334155')
    ax.set_xlim(0, max_v * 1.28)
    ax.set_xlabel(T('用户数（含点赞）', 'Users (incl. likes)'), fontsize=7, color='#475569')
    ax.set_title(T('(c) 认知维度分布（二级属性）', '(c) Distribution of Cognitive Dimensions'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig2c_attr2_bar')
    plt.close(fig)


# --- Fig 2d: Behavioural intentions bar ---
def fig2d_intent_bar():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    s = build_series(INTENT_COLS_MAIN, aggregates)
    colors = sns.color_palette("crest", n_colors=len(s))
    bars = ax.barh(s.index, s.values, color=colors, edgecolor='white',
                   linewidth=0.8, height=0.62)
    max_v = s.values.max()
    for bar, val in zip(bars, s.values):
        pct = val / TOTAL_USERS * 100
        ax.text(bar.get_width() + max_v * 0.018, bar.get_y() + bar.get_height() / 2,
                f'{val:,} ({pct:.1f}%)', va='center', fontsize=6, color='#334155')
    ax.set_xlim(0, max_v * 1.28)
    ax.set_xlabel(T('用户数（含点赞）', 'Users (incl. likes)'), fontsize=7, color='#475569')
    ax.set_title(T('(d) 行为意向分布', '(d) Distribution of Behavioural Intentions'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig2d_intent_bar')
    plt.close(fig)


# --- Fig 2e: Sentiment × Cognition heatmap ---
def fig2e_sentiment_attr2_heatmap():
    fig, ax = plt.subplots(figsize=(9.5, 2.3))
    df = build_cross_table(SENTIMENT_COLS, ATTR2_COLS_MAIN, sent_x_attr2, normalize='rows')
    sns.heatmap(
        df, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': T('占比 (%)', '%'), 'aspect': 20},
        vmin=0, vmax=df.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title(T('(e) 情感 × 认知维度（行百分比）', '(e) Sentiment × Cognitive Dimensions (row %)'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel(T('认知维度', 'Cognitive Dimension'), fontsize=7)
    ax.set_ylabel(T('情感倾向', 'Sentiment'), fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig2e_sentiment_attr2_heatmap')
    plt.close(fig)


# --- Fig 2f: Attr1 × Cognition heatmap ---
def fig2f_attr1_attr2_heatmap():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    df = build_cross_table(ATTR1_COLS_MAIN, ATTR2_COLS_MAIN, attr1_x_attr2, normalize='rows')
    sns.heatmap(
        df, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': T('占比 (%)', '%'), 'aspect': 25},
        vmin=0, vmax=df.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title(T('(f) 一级属性 × 认知维度（行百分比）', '(f) Primary Attributes × Cognitive Dimensions (row %)'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel(T('认知维度', 'Cognitive Dimension'), fontsize=7)
    ax.set_ylabel(T('一级属性', 'Primary Attribute'), fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig2f_attr1_attr2_heatmap')
    plt.close(fig)


# ============================================================
# 8. FIGURE FUNCTIONS — Argument 3: Geographic Distribution
# ============================================================

# --- Fig 3a: Province total bar ---
def fig3a_province_bar():
    fig, ax = plt.subplots(figsize=(8, 9.5))
    ordered = [p for p in PROVINCE_ORDER if p in province_aggregates]
    values = [province_aggregates[p] for p in ordered]
    labels = [province_display(p) for p in ordered]
    reg_colors = [C_REGION[get_region(p)] for p in ordered]

    bars = ax.barh(labels, values, color=reg_colors, edgecolor='white',
                   linewidth=0.5, height=0.7)
    max_v = max(values)
    for bar, val in zip(bars, values):
        pct = val / sum(values) * 100
        ax.text(bar.get_width() + max_v * 0.015, bar.get_y() + bar.get_height() / 2,
                f'{val:,}  ({pct:.1f}%)', va='center', fontsize=5.5, color='#334155')

    ax.set_xlim(0, max_v * 1.24)
    ax.set_xlabel(T('评论数', 'Number of Comments'), fontsize=7, color='#475569')
    ax.set_title(T('(a) 各省/地区评论分布', '(a) Comment Distribution by Province/Region'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(axis='y', labelsize=6.5)

    legend_items = [
        Patch(facecolor=v, label=region_label(k)) for k, v in C_REGION.items()
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=5.5,
              frameon=True, ncol=4, edgecolor='#d1d5db', fancybox=False)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig3a_province_bar')
    plt.close(fig)


# --- Fig 3b: Province sentiment stacked bar (Top 20) ---
def fig3b_province_sentiment():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    top_n = 20
    top_provs = prov_df.head(top_n)['province'].values
    sent_labels = [L(c) for c in SENTIMENT_COLS]

    data = {}
    for sc, sl in zip(SENTIMENT_COLS, sent_labels):
        data[sl] = [prov_x_sent.get((prov, sc), 0) for prov in top_provs]
    row_labels = [province_display(p) for p in top_provs]
    df_stack = pd.DataFrame(data, index=row_labels)
    df_stack = df_stack.div(df_stack.sum(axis=1), axis=0) * 100

    colors = [C_SENTIMENT.get(sl, '#8c8c8c') for sl in sent_labels]
    bottom = np.zeros(top_n)
    for col, color in zip(df_stack.columns, colors):
        ax.barh(df_stack.index, df_stack[col], left=bottom, color=color,
                edgecolor='white', linewidth=0.4, height=0.68, label=col)
        for j, (val, b) in enumerate(zip(df_stack[col], bottom)):
            if val > 6:
                txt_color = 'white' if color in ('#c44e52', '#5a9e6f') else '#2c3e50'
                ax.text(b + val / 2, j, f'{val:.1f}%', ha='center', va='center',
                        fontsize=5.5, fontweight='bold', color=txt_color)
        bottom += df_stack[col].values

    ax.set_xlabel(T('占比 (%)', 'Percentage (%)'), fontsize=7, color='#475569')
    ax.set_title(T('(b) Top 20 省份情感构成', '(b) Sentiment Composition of Top 20 Provinces'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.legend(loc='lower right', fontsize=6, frameon=True, ncol=3,
              edgecolor='#d1d5db', fancybox=False)
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig3b_province_sentiment')
    plt.close(fig)


# --- Fig 3c: Province × Cognition heatmap (Top 15) ---
def fig3c_province_attr2_heatmap():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    top_n = 15
    top_provs = prov_df.head(top_n)['province'].values
    attr2_labels = [L(c) for c in ATTR2_COLS_MAIN]

    mat = np.zeros((top_n, len(ATTR2_COLS_MAIN)))
    for i, prov in enumerate(top_provs):
        total = province_aggregates[prov]
        for j, ac in enumerate(ATTR2_COLS_MAIN):
            mat[i, j] = prov_x_attr2.get((prov, ac), 0) / total * 100
    row_labels = [province_display(p) for p in top_provs]
    df = pd.DataFrame(mat, index=row_labels, columns=attr2_labels)

    sns.heatmap(
        df, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': T('占比 (%)', '%'), 'aspect': 25},
        vmin=0, vmax=df.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title(T('(c) 认知维度省际分布（Top 15，行百分比）',
                   '(c) Cognitive Dimensions by Province (Top 15, row %)'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel(T('认知维度', 'Cognitive Dimension'), fontsize=7)
    ax.set_ylabel(T('省份', 'Province'), fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig3c_province_attr2_heatmap')
    plt.close(fig)


# --- Fig 3d: Province × Intention heatmap (Top 15) ---
def fig3d_province_intent_heatmap():
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    top_n = 15
    top_provs = prov_df.head(top_n)['province'].values
    intent_labels = [L(c) for c in INTENT_COLS_MAIN]

    mat = np.zeros((top_n, len(INTENT_COLS_MAIN)))
    for i, prov in enumerate(top_provs):
        total = province_aggregates[prov]
        for j, ic in enumerate(INTENT_COLS_MAIN):
            mat[i, j] = prov_x_intent.get((prov, ic), 0) / total * 100
    row_labels = [province_display(p) for p in top_provs]
    df = pd.DataFrame(mat, index=row_labels, columns=intent_labels)

    sns.heatmap(
        df, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': T('占比 (%)', '%'), 'aspect': 25},
        vmin=0, vmax=df.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title(T('(d) 行为意向省际分布（Top 15，行百分比）',
                   '(d) Behavioural Intentions by Province (Top 15, row %)'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel(T('行为意向', 'Behavioural Intention'), fontsize=7)
    ax.set_ylabel(T('省份', 'Province'), fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig3d_province_intent_heatmap')
    plt.close(fig)


# --- Fig 3e: Province cognitive radar (8 representative provinces) ---
def fig3e_province_radar():
    rep_provs = list(prov_df.head(6)['province'])
    for p in ['香港特别行政区', '台湾省', '澳门特别行政区']:
        if p in province_aggregates and p not in rep_provs:
            rep_provs.append(p)
            if len(rep_provs) >= 8:
                break
    rep_provs = rep_provs[:8]

    attr2_labels = [L(c) for c in ATTR2_COLS_MAIN]
    n_vars = len(ATTR2_COLS_MAIN)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#c44e52', '#4c72b0', '#55a868', '#dd8452',
              '#937860', '#64b5cd', '#da8bc3', '#8c8c8c']

    max_val_all = 0
    for idx, prov in enumerate(rep_provs):
        total = province_aggregates[prov]
        values = [prov_x_attr2.get((prov, ac), 0) / total * 100 for ac in ATTR2_COLS_MAIN]
        values += values[:1]
        max_val_all = max(max_val_all, max(values[:-1]))
        label = province_display(prov)
        ax.fill(angles, values, alpha=0.06, color=colors[idx])
        ax.plot(angles, values, 'o-', linewidth=1.5, color=colors[idx],
                label=label, markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attr2_labels, fontsize=6.5)
    ax.set_ylim(0, max_val_all * 1.3)
    ax.set_yticks([])
    ax.spines['polar'].set_linewidth(0.4)
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_title(T('(e) 代表性省份认知维度对比',
                   '(e) Cognitive Dimension Comparison — Representative Provinces'),
                 fontsize=9, fontweight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.08),
              fontsize=6.5, frameon=True, edgecolor='#d1d5db', fancybox=False)
    fig.tight_layout(pad=2.5)
    save_figure(fig, 'fig3e_province_radar')
    plt.close(fig)


# --- Fig 3f: Cognition scatter — Cultural Philosophy vs Combat Skills ---
def fig3f_cognition_scatter():
    fig, ax = plt.subplots(figsize=(8, 7.5))
    # Exclude HK/Macau/Taiwan for better mainland spread
    provs = [p for p in province_aggregates
             if province_aggregates[p] >= 200 and get_region(p) != 'HK/Macau/Taiwan']
    x_vals, y_vals, sizes, labels_list, colors_list = [], [], [], [], []

    for prov in provs:
        total = province_aggregates[prov]
        x = prov_x_attr2.get((prov, '属性二_实战技术'), 0) / total * 100
        y = prov_x_attr2.get((prov, '属性二_文化哲学'), 0) / total * 100
        x_vals.append(x)
        y_vals.append(y)
        sizes.append(np.sqrt(total) * 25)
        labels_list.append(province_display(prov))
        colors_list.append(C_REGION[get_region(prov)])

    ax.scatter(x_vals, y_vals, s=sizes, c=colors_list, alpha=0.72,
               edgecolors='white', linewidth=0.8)

    x_pad = (max(x_vals) - min(x_vals)) * 0.28 or 1.5
    y_pad = (max(y_vals) - min(y_vals)) * 0.28 or 1.5
    x_lim = (min(x_vals) - x_pad, max(x_vals) + x_pad)
    y_lim = (min(y_vals) - y_pad, max(y_vals) + y_pad)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    try:
        from adjustText import adjust_text
        texts = [ax.annotate(lab, (x_vals[i], y_vals[i]), fontsize=5.5,
                             ha='center', va='center', color='#1e293b')
                 for i, lab in enumerate(labels_list)]
        adjust_text(texts, ax=ax, expand_points=(1.2, 1.2), expand_text=(1.15, 1.15),
                    force_text=0.25, force_points=0.25, lim=300)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
    except ImportError:
        for i, lab in enumerate(labels_list):
            ax.annotate(lab, (x_vals[i], y_vals[i]), fontsize=5.5,
                        ha='center', va='bottom', color='#1e293b',
                        xytext=(0, 6), textcoords='offset points')

    x_mid, y_mid = np.median(x_vals), np.median(y_vals)
    ax.axvline(x_mid, color='#94a3b8', linestyle='--', alpha=0.35, linewidth=0.7)
    ax.axhline(y_mid, color='#94a3b8', linestyle='--', alpha=0.35, linewidth=0.7)

    # Quadrant labels
    ax.text(x_lim[1] * 0.98, y_lim[1] * 0.97,
            T(' ', 'High Combat, High Culture'),
            ha='right', va='top', fontsize=5.5, color='#94a3b8', style='italic')
    ax.text(x_lim[0] + (x_lim[1]-x_lim[0]) * 0.02, y_lim[1] * 0.97,
            T(' ', 'Low Combat, High Culture'),
            ha='left', va='top', fontsize=5.5, color='#94a3b8', style='italic')

    ax.set_xlabel(T('实战技术关注度 (%)', 'Combat Skills Focus (%)'), fontsize=7, color='#475569')
    ax.set_ylabel(T('文化哲学关注度 (%)', 'Cultural Philosophy Focus (%)'), fontsize=7, color='#475569')
    ax.set_title(T('(f) 各省认知偏好分布\n（气泡大小 ∝ 评论量，虚线=中位数）',
                   '(f) Cognitive Preference by Province\n(bubble size ∝ comments, dashed = median)'),
                 fontsize=9, fontweight='bold', pad=12)
    legend_items = [Patch(facecolor=v, label=region_label(k))
                    for k, v in C_REGION.items() if k != 'HK/Macau/Taiwan']
    ax.legend(handles=legend_items, loc='upper left', fontsize=5.5,
              frameon=True, edgecolor='#d1d5db', fancybox=False)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig3f_cognition_scatter')
    plt.close(fig)


# --- Fig 3g: China choropleth map (pyecharts + playwright) ---
def fig3g_china_map():
    """Generate China province heatmap as static image. Requires playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('  [SKIP] fig3g: playwright not installed (pip install playwright && playwright install chromium)')
        return

    import json, re, ssl, urllib.request

    data_pairs = [[prov_cn, count] for prov_cn, count in province_aggregates.items()]
    data_json = json.dumps(data_pairs, ensure_ascii=False)

    max_val = max(province_aggregates.values())
    min_val = min(province_aggregates.values())

    # Download ECharts JS if needed
    js_dir = OUTPUT_DIR / 'js'
    js_dir.mkdir(parents=True, exist_ok=True)
    echarts_js = js_dir / 'echarts.min.js'
    china_js = js_dir / 'china.js'

    ctx_ssl = ssl.create_default_context()
    ctx_ssl.check_hostname = False
    ctx_ssl.verify_mode = ssl.CERT_NONE

    for fpath, url in [
        (echarts_js, 'https://assets.pyecharts.org/assets/v6/echarts.min.js'),
        (china_js, 'https://assets.pyecharts.org/assets/v6/maps/china.js'),
    ]:
        if not fpath.exists():
            try:
                data = urllib.request.urlopen(url, context=ctx_ssl, timeout=30).read()
                fpath.write_bytes(data)
            except Exception as e:
                print(f'  [WARN] Cannot download {url}: {e}')
                return

    echarts_src = echarts_js.read_text(encoding='utf-8')
    china_src = china_js.read_text(encoding='utf-8')

    map_title = T('(g) 太极拳相关评论地理分布', '(g) Geographic Distribution of Tai Chi Comments in China')
    high_low = ['高', '低'] if LANG == 'zh' else ['High', 'Low']

    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>China Map</title>
<style>body{{margin:0;background:#fff;font-family:"Microsoft YaHei",Arial,sans-serif;}}</style></head>
<body><div id="chart" style="width:1600px;height:1200px;"></div>
<script>{echarts_src}</script>
<script>{china_src}</script>
<script>
var c = echarts.init(document.getElementById('chart'), null, {{renderer:'canvas'}});
c.setOption({{
    title:{{text:'{map_title}',left:'center',top:15,
        textStyle:{{fontSize:20,fontWeight:'bold',color:'#1e293b',fontFamily:'Microsoft YaHei,Arial'}}}},
    visualMap:{{type:'continuous',min:{min_val},max:{max_val},left:28,top:'center',
        orient:'vertical',text:{high_low},
        textStyle:{{color:'#475569',fontSize:14,fontWeight:'bold'}},
        itemWidth:24,itemHeight:200,
        inRange:{{color:['#f1f5f9','#cbd5e1','#94a3b8','#64748b','#475569','#334155','#1e293b','#0f172a','#020617']}}}},
    series:[{{name:'Comments',type:'map',map:'china',roam:false,
        label:{{show:false}},
        itemStyle:{{borderColor:'rgba(100,116,139,0.2)',borderWidth:1.2,areaColor:'#f8fafc'}},
        emphasis:{{label:{{show:true,fontSize:16,fontWeight:'bold',color:'#0f172a'}},
            itemStyle:{{borderColor:'#334155',borderWidth:2.5}}}},
        data:{data_json}}}]}});
</script></body></html>'''

    html_path = OUTPUT_DIR / '_fig3g_tmp.html'
    html_path.write_text(html, encoding='utf-8')

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1200})
        page.goto(f'file:///{html_path.as_posix()}', wait_until='load')
        page.wait_for_timeout(2500)
        png_path = OUTPUT_DIR / 'fig3g_china_map.png'
        page.screenshot(path=str(png_path), full_page=True)
        browser.close()

    if html_path.exists():
        html_path.unlink()
    print(f'  Saved: fig3g_china_map.png (raster, 1600×1200)')


# ============================================================
# 9. EXPLORATORY FIGURES — Temporal, engagement, cross-dimension
# ============================================================

# --- Fig 4a: Monthly comment volume with sentiment overlay ---
def fig4a_temporal_trend():
    if not monthly_total:
        print('  [SKIP] fig4a: no temporal data')
        return
    months = sorted(monthly_total.keys())
    totals = [monthly_total[m] for m in months]
    sent_keys = [L(c) for c in SENTIMENT_COLS]
    sent_stacked = {}
    for sc, sk in zip(SENTIMENT_COLS, sent_keys):
        sent_stacked[sk] = [monthly_sent.get((m, sc), 0) for m in months]

    # Nature-style sentiment palette — lower saturation, unified family feel
    _PAL_SENT = {
        '肯定/赞美': '#8BCF8B',   # green_3  — affirmation
        '诋毁/谩骂': '#B64342',   # red_strong — criticism
        '客观中立': '#767676',   # neutral_mid — neutral
        'Affirmation': '#8BCF8B',
        'Criticism': '#B64342',
        'Neutral': '#767676',
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 5.8), sharex=True,
                                    gridspec_kw={'height_ratios': [1.8, 1]})

    # Top: stacked area of sentiment
    colors_s = [_PAL_SENT.get(k, '#767676') for k in sent_keys]
    ax1.stackplot(range(len(months)), *sent_stacked.values(),
                  labels=sent_stacked.keys(), colors=colors_s, alpha=0.85)
    ax1.set_ylabel(T('评论数', 'Comments'), fontsize=7, color='#475569')
    ax1.legend(loc='upper left', fontsize=5.5, frameon=False, ncol=3)
    ax1.set_title(T('评论月度趋势与情感构成', 'Monthly Comment Volume & Sentiment'),
                  fontsize=9, fontweight='bold', pad=10)
    ax1.tick_params(labelsize=6)

    # Bottom: total bar with value-driven colour gradient
    norm = mpl.colors.Normalize(vmin=min(totals), vmax=max(totals))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'blue_grad', ['#B4C0E4', '#7884B4', '#3775BA', '#0F4D92'])
    bar_colors = [cmap(norm(v)) for v in totals]
    ax2.bar(range(len(months)), totals, color=bar_colors, width=0.72,
            edgecolor='white', linewidth=0.3)
    ax2.set_ylabel(T('总评论数', 'Total'), fontsize=7, color='#475569')
    ax2.set_xlabel(T('月份', 'Month'), fontsize=7, color='#475569')

    # Dashed mean reference line
    mean_val = np.mean(totals)
    ax2.axhline(mean_val, color='#767676', linestyle='--', linewidth=0.6, alpha=0.7)
    ax2.text(len(months) - 0.5, mean_val,
             T(f'均值 {mean_val:,.0f}', f'Mean {mean_val:,.0f}'),
             va='bottom', ha='right', fontsize=5, color='#767676')

    # Annotate top-3 peaks
    peak_idx = np.argsort(totals)[-3:]
    for pi in peak_idx:
        ax2.annotate(f'{totals[pi]:,}', (pi, totals[pi]),
                     textcoords='offset points', xytext=(0, 3),
                     fontsize=5, ha='center', va='bottom', color='#272727')

    # X-axis: show every Nth label to avoid crowding
    step = max(1, len(months) // 12)
    tick_positions = list(range(0, len(months), step))
    tick_labels = [months[i] for i in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=5.5)
    ax2.tick_params(labelsize=6)

    # Panel labels (Nature-style: bold lowercase near top-left)
    for ax_i, lbl in zip((ax1, ax2), ('a', 'b')):
        ax_i.text(-0.05, 1.04, lbl, transform=ax_i.transAxes,
                  fontsize=10, fontweight='bold', color='#272727',
                  ha='left', va='bottom')

    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig4a_temporal_trend')
    plt.close(fig)


# --- Fig 4b: Engagement — avg likes per comment by cognitive dimension ---
def fig4b_engagement_attr2():
    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = [L(c) for c in ATTR2_COLS_MAIN]
    likes_vals = [category_likes_raw[c] for c in ATTR2_COLS_MAIN]
    count_vals = [aggregates[c] for c in ATTR2_COLS_MAIN]
    avg_likes = [l / max(c, 1) for l, c in zip(likes_vals, count_vals)]
    total_vals = [aggregates[c] for c in ATTR2_COLS_MAIN]

    # Sort by avg likes descending
    order = np.argsort(avg_likes)
    labels_ord = [labels[i] for i in order]
    avg_ord = [avg_likes[i] for i in order]
    total_ord = [total_vals[i] for i in order]

    # Value-driven gradient: lighter → darker = lower → higher engagement
    norm = mpl.colors.Normalize(vmin=min(avg_ord), vmax=max(avg_ord))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'engage_grad', ['#B4C0E4', '#7884B4', '#3775BA', '#0F4D92'])
    bar_colors = [cmap(norm(v)) for v in avg_ord]

    bars = ax.barh(labels_ord, avg_ord, color=bar_colors, edgecolor='white',
                   linewidth=0.6, height=0.62)
    max_v = max(avg_ord)
    for bar, avg, tot in zip(bars, avg_ord, total_ord):
        ax.text(bar.get_width() + max_v * 0.02, bar.get_y() + bar.get_height() / 2,
                f'{avg:.1f}  (n={tot:,})', va='center', fontsize=5.5, color='#334155')

    ax.set_xlim(0, max_v * 1.22)
    ax.set_xlabel(T('平均点赞数', 'Avg. Likes per Comment'), fontsize=7, color='#475569')
    ax.set_title(T('各认知维度平均点赞数', 'Avg. Likes per Comment by Cognitive Dimension'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.tick_params(labelsize=6.5)

    # Panel label
    ax.text(-0.06, 1.04, 'c', transform=ax.transAxes,
            fontsize=10, fontweight='bold', color='#272727',
            ha='left', va='bottom')

    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig4b_engagement_attr2')
    plt.close(fig)


# --- Fig 4c: Sentiment × Intention heatmap ---
def fig4c_sentiment_intent_heatmap():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    df = build_cross_table(SENTIMENT_COLS, INTENT_COLS_MAIN, sent_x_intent, normalize='rows')
    sns.heatmap(
        df, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': T('占比 (%)', '%'), 'aspect': 20},
        vmin=0, vmax=df.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title(T('(c) 情感 × 行为意向（行百分比）', '(c) Sentiment × Behavioural Intentions (row %)'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel(T('行为意向', 'Behavioural Intention'), fontsize=7)
    ax.set_ylabel(T('情感倾向', 'Sentiment'), fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig4c_sentiment_intent_heatmap')
    plt.close(fig)


# --- Fig 4e: Top contributing provinces per cognitive dimension ---
def fig4e_top_provinces_per_dim():
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flatten()
    dim_labels = [L(c) for c in ATTR2_COLS_MAIN]

    for idx, (ac, dlab) in enumerate(zip(ATTR2_COLS_MAIN, dim_labels)):
        ax = axes[idx]
        prov_vals = {}
        for prov in province_aggregates:
            total = province_aggregates[prov]
            if total < 200:
                continue
            pct = prov_x_attr2.get((prov, ac), 0) / total * 100
            prov_vals[prov] = pct
        top5 = sorted(prov_vals.items(), key=lambda x: -x[1])[:5]
        names = [province_display(p) for p, _ in top5]
        vals = [v for _, v in top5]
        colors = [C_REGION[get_region(p)] for p, _ in top5]

        bars = ax.barh(names, vals, color=colors, edgecolor='white',
                       linewidth=0.6, height=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + max(vals) * 0.03, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=5, color='#334155')
        ax.set_xlim(0, max(vals) * 1.18)
        ax.set_title(dlab, fontsize=7, fontweight='bold', color='#1e293b')
        ax.tick_params(labelsize=5.5)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)

    fig.suptitle(T('(e) 各认知维度Top 5省份占比', '(e) Top 5 Provinces per Cognitive Dimension'),
                 fontsize=10, fontweight='bold', y=1.01)
    fig.tight_layout(pad=2.5)
    save_figure(fig, 'fig4e_top_provinces_per_dim')
    plt.close(fig)


# --- Fig 4f: Attr1 × Intention heatmap ---
def fig4f_attr1_intent_heatmap():
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    df = build_cross_table(ATTR1_COLS_MAIN, INTENT_COLS_MAIN, attr1_x_intent, normalize='rows')
    sns.heatmap(
        df, annot=True, fmt='.1f', cmap=C_HEATMAP, ax=ax,
        linewidths=0.6, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': T('占比 (%)', '%'), 'aspect': 25},
        vmin=0, vmax=df.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax.set_title(T('(f) 一级属性 × 行为意向（行百分比）', '(f) Primary Attributes × Behavioural Intentions (row %)'),
                 fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel(T('行为意向', 'Behavioural Intention'), fontsize=7)
    ax.set_ylabel(T('一级属性', 'Primary Attribute'), fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig4f_attr1_intent_heatmap')
    plt.close(fig)


# ============================================================
# 10. Summary statistics export
# ============================================================
def export_summary():
    lines = []
    lines.append('=' * 70)
    lines.append('Tai Chi Cultural Heritage Cognitive Analysis — Summary')
    lines.append('=' * 70)
    lines.append(f'Total comments (real users): {total_rows:,}')
    lines.append(f'Total likes: {total_likes:,}')
    lines.append(f'Total users (real + likes): {TOTAL_USERS:,}')
    lines.append(f'Comments with IP location: {sum(province_aggregates.values()):,} '
                 f'({sum(province_aggregates.values())/total_rows*100:.1f}%)')
    lines.append(f'Provinces/regions with data: {len(province_aggregates)}')
    lines.append('')

    lines.append('--- Sentiment ---')
    for c in SENTIMENT_COLS:
        pct = aggregates[c] / TOTAL_USERS * 100
        lines.append(f'  {L(c)}: {aggregates[c]:,} ({pct:.1f}%)')

    lines.append('')
    lines.append('--- Cognitive Dimensions (Attr2) ---')
    for c in ATTR2_COLS_MAIN:
        pct = aggregates[c] / TOTAL_USERS * 100
        lines.append(f'  {L(c)}: {aggregates[c]:,} ({pct:.1f}%)')

    lines.append('')
    lines.append('--- Behavioural Intentions ---')
    for c in INTENT_COLS_MAIN:
        pct = aggregates[c] / TOTAL_USERS * 100
        lines.append(f'  {L(c)}: {aggregates[c]:,} ({pct:.1f}%)')

    lines.append('')
    lines.append('--- Top 10 Provinces ---')
    for i, row in prov_df.head(10).iterrows():
        pct = row['total'] / sum(province_aggregates.values()) * 100
        name = province_display(row['province'])
        lines.append(f'  {i+1}. {name}: {row["total"]:,} ({pct:.1f}%)')

    lines.append('')
    lines.append('--- Bottom 5 Provinces ---')
    for i, (idx, row) in enumerate(prov_df.tail(5).iterrows()):
        pct = row['total'] / sum(province_aggregates.values()) * 100
        name = province_display(row['province'])
        lines.append(f'  {len(prov_df)-4+i}. {name}: {row["total"]:,} ({pct:.1f}%)')

    path = OUTPUT_DIR / 'summary_statistics.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  Saved: summary_statistics.txt')


# ============================================================
# 10. Main
# ============================================================
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('Nature-Quality Figure Generation')
    print(f'Language: {LANG.upper()}  |  Output: {OUTPUT_DIR}')
    print('=' * 60)

    print('\n--- Argument 2: Cognition & Value Identity ---')
    fig2a_sentiment_donut()
    fig2b_attr1_bar()
    fig2c_attr2_bar()
    fig2d_intent_bar()
    fig2e_sentiment_attr2_heatmap()
    fig2f_attr1_attr2_heatmap()

    print('\n--- Argument 3: Geographic Distribution ---')
    fig3a_province_bar()
    fig3b_province_sentiment()
    fig3c_province_attr2_heatmap()
    fig3d_province_intent_heatmap()
    fig3e_province_radar()
    fig3f_cognition_scatter()
    fig3g_china_map()

    print('\n--- Exploratory: Temporal, Engagement & Cross-Dimension ---')
    fig4a_temporal_trend()
    fig4b_engagement_attr2()
    fig4c_sentiment_intent_heatmap()
    fig4e_top_provinces_per_dim()
    fig4f_attr1_intent_heatmap()

    print('\n--- Summary ---')
    export_summary()

    print('\n' + '=' * 60)
    print(f'Done. All figures saved to: {OUTPUT_DIR}')
    print(f'Total: 18 figures (13 core + 5 exploratory)')
    print('=' * 60)
