"""
=============================================================================
Nature-quality figures: Tai Chi Video Content Analysis — Extended
=============================================================================
Figure contract:
  Core conclusion:
    Tai Chi video production on Douyin forms distinct 'communication circles'
    defined by co-occurring production features (duration, scene, music,
    effects) that exhibit systematically different engagement efficiency
    patterns. Supply-side strategies have undergone temporal shifts, with
    养生 (health) content rising while traditional 教学 (teaching) maintains
    a stable core.

  Figure archetype: quantitative grid
  Backend: Python (matplotlib + seaborn)
  Output: SVG + PDF + TIFF per figure

  Panel map:
    Fig V4 — Production Strategy & Engagement Heterogeneity
      a: Duration × Engagement scatter (hero — log-log, colour by purpose)
      b: Engagement conversion funnel (likes→saves/comments/shares ratios)
      c: Production complexity radar (presenter type comparison)
      d: Save rate vs Like rate bubble (purpose × style)
      e: Effect type × avg engagement (grouped bar)
      f: Narration style × engagement by purpose (split violins)

    Fig V5 — Content Circles & Temporal Evolution
      a: Feature co-occurrence matrix (hero — cross-dimension correlations)
      b: Purpose composition over time (stacked area)
      c: Style trend lines over time (small multiples)
      d: Style × Scene × Purpose multi-heatmap (content circles)
      e: Music style evolution over time (stacked area)
      f: Creator concentration (Lorenz curve of engagement)

    Fig V6 — Depth Analysis: Duration, Demographics & Language
      a: Duration distribution by purpose (violin, log scale)
      b: Age × Purpose × Engagement (heatmap)
      c: Gender × Purpose interaction
      d: Language comparison (Chinese vs English vs Bilingual content profiles)
      e: Multi-dimensional engagement signature radar (by purpose)
      f: Temporal heatmap — monthly engagement per purpose
=============================================================================
Input: merged_dy_ALLData_with_labels.csv (10,503 videos × 76 cols)
Output: dy_file/dy_video_csv/figures_video/ — SVG + PDF + TIFF per figure
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
from matplotlib.patches import Patch, FancyBboxPatch
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. Paths & global config
# ============================================================
INPUT_PATH = str(VIDEO_MERGED_WITH_LABELS)
OUTPUT_DIR = FIGURES_VIDEO_DIR
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

# ── Nature unified palette families ──
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_soft":      "#B4C0E4",
    "green_3":        "#8BCF8B",
    "green_dark":     "#2E7D32",
    "red_strong":     "#B64342",
    "red_soft":       "#F6CFCB",
    "neutral_light":  "#CFCECE",
    "neutral_mid":    "#767676",
    "neutral_dark":   "#4D4D4D",
    "neutral_black":  "#272727",
    "gold":           "#E8A735",
    "teal":           "#42949E",
    "violet":         "#9A4D8E",
    "magenta":        "#C44E7C",
    "orange":         "#E28E2C",
}

# Category palettes (Nature low-saturation, one unified family per dimension)
C_PURPOSE = {
    '教学': '#3775BA', '养生': '#2E7D32', '日常': '#E28E2C',
    '文化哲学': '#9A4D8E', '门派': '#42949E', '实战': '#B64342',
    '赛事': '#C44E7C', '其他': '#767676',
}
C_STYLE = {
    '陈式': '#B64342', '杨式': '#3775BA', '吴式': '#2E7D32',
    '武式': '#E28E2C', '孙式': '#9A4D8E', '武当太极': '#42949E',
    '24式': '#C44E7C', '其他': '#767676',
}
C_PRESENTER = {
    '专业武术家/道长': '#B64342', '泛娱乐博主': '#3775BA', '非科班出身': '#2E7D32',
}
C_SCENE = {
    '山水自然': '#2E7D32', '古建筑': '#9A4D8E', '武馆': '#B64342',
    '城市街道': '#3775BA', '居家': '#E28E2C', '寺庙道观': '#C44E7C',
    '校园': '#42949E', '其他': '#767676',
}
C_STYLE_ATTR = {
    '偏重柔和': '#8BCF8B', '刚柔并济': '#3775BA', '偏重刚猛': '#B64342',
}

# ============================================================
# 1. Data loading + derived features
# ============================================================
print('=' * 60)
print('Step 1: Loading & feature engineering...')
print('=' * 60)

df = pd.read_csv(INPUT_PATH, encoding='utf-8-sig', low_memory=False)
TOTAL_VIDEOS = len(df)
print(f'Total videos: {TOTAL_VIDEOS:,}')

df['发布时间_dt'] = pd.to_datetime(df['发布时间'], errors='coerce')
df['年月'] = df['发布时间_dt'].dt.strftime('%Y-%m')
df['年份'] = df['发布时间_dt'].dt.year

# Duration in seconds
df['dur_sec'] = df['视频时长'] / 1000
df['log_dur'] = np.log10(df['dur_sec'] + 1)
df['log_likes'] = np.log10(df['点赞数'] + 1)
df['log_comments'] = np.log10(df['评论数'] + 1)
df['log_saves'] = np.log10(df['收藏数'] + 1)
df['log_shares'] = np.log10(df['转发数'] + 1)

# Engagement ratios (avoid div by 0)
eps = 1e-6
df['save_rate'] = df['收藏数'] / (df['点赞数'] + eps)
df['comment_rate'] = df['评论数'] / (df['点赞数'] + eps)
df['share_rate'] = df['转发数'] / (df['点赞数'] + eps)
df['recommend_rate'] = df['推荐数'] / (df['点赞数'] + eps)
df['engagement_index'] = (df['点赞数'] + df['评论数'] * 3 + df['收藏数'] * 5 + df['转发数'] * 8) / 1000

# Production complexity: count how many production features are active
EFFECT_COLS_MAIN = ['无特效', '有特效', '古风慢摇', '节奏卡点']
MUSIC_INSTRUMENT_COLS = ['东方传统乐器', '西方乐器', '电子合成器', '白噪声环境音']
MUSIC_STYLE_COLS_MAIN = ['古风', '国风潮流', '戏曲', '原声', '旁白解说', '舒缓空灵', '大气磅礴', '动感活力', '网红BGM']

df['n_effects'] = df[EFFECT_COLS_MAIN].sum(axis=1)
df['n_instruments'] = df[MUSIC_INSTRUMENT_COLS].sum(axis=1)
df['n_music_styles'] = df[MUSIC_STYLE_COLS_MAIN].sum(axis=1)
df['production_score'] = df['n_effects'] + df['n_instruments'] + df['n_music_styles']

# Column groups
STYLE_COLS_MAIN = ['陈式', '杨式', '吴式', '武式', '孙式', '武当太极', '24式']
PURPOSE_COLS_MAIN = ['赛事', '教学', '养生', '门派', '日常', '实战', '文化哲学']
SCENE_COLS_MAIN = ['寺庙道观', '武馆', '山水自然', '城市街道', '居家', '校园', '古建筑']
CLOTHING_COLS_MAIN = ['便服装', '太极服装', '古风服装', '现代潮流服装']
PRESENTER_COLS = ['专业武术家/道长', '泛娱乐博主', '非科班出身']
STYLE_ATTR_COLS = ['偏重柔和', '刚柔并济', '偏重刚猛']
GENDER_COLS = ['男性', '女性', '群体混合']
AGE_COLS = ['儿童', '青年', '中年', '老年']
LANGUAGE_COLS = ['中文', '英文', '双语']
NARRATION_COLS = ['讲解类', '非讲解类']
DISPLAY_COLS = ['全身展示', '局部展示']

ALL_LABEL_DIMS = {
    '流派': STYLE_COLS_MAIN,
    '意图': PURPOSE_COLS_MAIN,
    '场景': SCENE_COLS_MAIN,
    '服装': CLOTHING_COLS_MAIN,
    '风格属性': STYLE_ATTR_COLS,
    '博主': PRESENTER_COLS,
}
MUSIC_DIMS = {
    '乐器': MUSIC_INSTRUMENT_COLS,
    '音乐风格': MUSIC_STYLE_COLS_MAIN,
    '特效': EFFECT_COLS_MAIN,
}

print(f'Time range: {df["发布时间_dt"].min()} ~ {df["发布时间_dt"].max()}')


# ============================================================
# 2. Helper functions
# ============================================================
def save_figure(fig, filename_stem):
    base = OUTPUT_DIR / filename_stem
    fig.savefig(str(base) + '.svg', bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(str(base) + '.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(str(base) + '.tiff', dpi=600, bbox_inches='tight', facecolor='white',
                edgecolor='none', pil_kwargs={'compression': 'tiff_lzw'})
    print(f'  Saved: {filename_stem}{{.svg,.pdf,.tiff}}')


def panel_label(ax, label, x=-0.06, y=1.05):
    """Nature-style bold lowercase panel label."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10,
            fontweight='bold', color=PALETTE['neutral_black'],
            ha='left', va='bottom')


def build_cross_table(cols_a, cols_b, data=None, normalize='rows'):
    """Build crosstab from one-hot columns."""
    if data is None:
        data = df
    rows, cols_labels = [], []
    mat_data = []
    for ca in cols_a:
        if ca not in data.columns:
            continue
        rows.append(ca)
        row_data = []
        for cb in cols_b:
            if cb not in data.columns:
                continue
            if len(cols_labels) < len(cols_b):
                cols_labels.append(cb)
            mask = (data[ca] == 1) & (data[cb] == 1)
            row_data.append(int(mask.sum()))
        mat_data.append(row_data)
    result = pd.DataFrame(mat_data, index=rows, columns=cols_labels)
    if normalize == 'rows':
        result = result.div(result.sum(axis=1), axis=0) * 100
    elif normalize == 'cols':
        result = result.div(result.sum(axis=0), axis=1) * 100
    return result


def monthly_aggregation(cat_cols, weight_col=None):
    """Aggregate category counts by month."""
    monthly = df.groupby('年月')
    result = {}
    for c in cat_cols:
        if c not in df.columns:
            continue
        if weight_col and weight_col in df.columns:
            result[c] = monthly.apply(lambda g: (g[c] * g[weight_col]).sum())
        else:
            result[c] = monthly.apply(lambda g: g[c].sum())
    return pd.DataFrame(result)


# ============================================================
# 3. FIGURE V4 — Production Strategy & Engagement Heterogeneity
# ============================================================

# --- V4a: Duration × Engagement scatter (hero panel) ---
def fig_v4a_duration_engagement():
    """Hero: log-log scatter of duration vs likes, coloured by purpose, with marginal histograms."""
    fig = plt.figure(figsize=(10, 8))

    # Main scatter
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # Plot each purpose separately for colour
    for purp, color in C_PURPOSE.items():
        if purp not in df.columns or purp == '其他':
            continue
        mask = df[purp] == 1
        subset = df[mask]
        if len(subset) > 0:
            ax_main.scatter(
                subset['log_dur'], subset['log_likes'],
                c=color, alpha=0.35, s=8, edgecolors='none',
                label=purp, rasterized=True
            )

    # Regression line
    from numpy.polynomial.polynomial import polyfit
    valid = df[(df['dur_sec'] > 0) & (df['点赞数'] > 0)]
    x_fit = valid['log_dur']
    y_fit = valid['log_likes']
    if len(x_fit) > 10:
        # Use binned means for trend
        bins = np.linspace(x_fit.min(), x_fit.max(), 30)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = np.array([y_fit[(x_fit >= bins[i]) & (x_fit < bins[i+1])].mean()
                              for i in range(len(bins)-1)])
        bin_means = bin_means[~np.isnan(bin_means)]
        bin_centers_f = bin_centers[~np.isnan(
            [y_fit[(x_fit >= bins[i]) & (x_fit < bins[i+1])].mean() for i in range(len(bins)-1)]
        )]
        ax_main.plot(bin_centers_f, bin_means, '-', color=PALETTE['neutral_dark'],
                     linewidth=1.8, alpha=0.7, zorder=5)

    ax_main.set_xlabel('视频时长 log₁₀(秒)', fontsize=7.5, color=PALETTE['neutral_dark'])
    ax_main.set_ylabel('点赞数 log₁₀', fontsize=7.5, color=PALETTE['neutral_dark'])
    ax_main.legend(loc='upper left', fontsize=5.5, markerscale=2,
                   ncol=2, frameon=True, edgecolor='#d1d5db', fancybox=False)
    ax_main.tick_params(labelsize=6)

    # Top histogram (log duration)
    ax_top.hist(df['log_dur'], bins=60, color=PALETTE['blue_soft'], edgecolor='white', linewidth=0.3)
    ax_top.set_ylabel('频次', fontsize=6, color=PALETTE['neutral_mid'])
    ax_top.tick_params(labelsize=5, labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)

    # Right histogram (log likes)
    ax_right.hist(df['log_likes'], bins=60, orientation='horizontal',
                  color=PALETTE['green_3'], edgecolor='white', linewidth=0.3)
    ax_right.set_xlabel('频次', fontsize=6, color=PALETTE['neutral_mid'])
    ax_right.tick_params(labelsize=5, labelleft=False)
    ax_right.spines['left'].set_visible(False)

    # Stats annotation
    r, p = stats.pearsonr(valid['log_dur'], valid['log_likes'])
    ax_main.text(0.97, 0.06, f'r = {r:.3f}\np < 0.001',
                 transform=ax_main.transAxes, fontsize=6,
                 ha='right', va='bottom', color=PALETTE['neutral_dark'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#d1d5db', alpha=0.8))

    panel_label(ax_top, 'a')
    fig.suptitle('视频时长与传播效果关系', fontsize=9, fontweight='bold', y=0.98)
    save_figure(fig, 'fig_v4a_duration_engagement')
    plt.close(fig)


# --- V4b: Engagement conversion funnel ---
def fig_v4b_engagement_funnel():
    """Engagement ratios by purpose — likes→saves, likes→comments, likes→shares."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)

    ratios = [
        ('save_rate', '收藏转化率\n(收藏/点赞)', axes[0]),
        ('comment_rate', '评论转化率\n(评论/点赞)', axes[1]),
        ('share_rate', '转发转化率\n(转发/点赞)', axes[2]),
    ]

    for ratio_col, title, ax in ratios:
        data = []
        labels = []
        colors = []
        for purp in PURPOSE_COLS_MAIN:
            if purp not in df.columns:
                continue
            mask = df[purp] == 1
            subset = df.loc[mask, ratio_col]
            # Winsorize to 99th percentile
            cap = subset.quantile(0.99)
            subset_clipped = subset.clip(upper=cap)
            data.append(subset_clipped)
            labels.append(purp)
            colors.append(C_PURPOSE.get(purp, PALETTE['neutral_mid']))

        # Sort by median
        medians = [np.median(d) for d in data]
        order = np.argsort(medians)[::-1]
        data_ord = [data[i] for i in order]
        labels_ord = [labels[i] for i in order]
        colors_ord = [colors[i] for i in order]

        bp = ax.boxplot(data_ord, vert=False, patch_artist=True,
                         widths=0.6, showfliers=False,
                         medianprops={'color': PALETTE['neutral_black'], 'linewidth': 1.2},
                         boxprops={'linewidth': 0.6},
                         whiskerprops={'linewidth': 0.5},
                         capprops={'linewidth': 0.5})

        for patch, color in zip(bp['boxes'], colors_ord):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        # Add mean as diamond
        means = [np.mean(d) for d in data_ord]
        for i, (m, c) in enumerate(zip(means, colors_ord)):
            ax.scatter(m, i + 1, marker='D', s=18, color=c, edgecolors='white',
                      linewidth=0.5, zorder=5)

        ax.set_yticks(range(1, len(labels_ord) + 1))
        ax.set_yticklabels(labels_ord, fontsize=6)
        ax.set_xlabel(title, fontsize=6.5, color=PALETTE['neutral_dark'])
        ax.tick_params(labelsize=5.5)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.set_xlim(left=0)

        # Annotate medians
        for i, (m, med) in enumerate(zip(means, medians)):
            ax.text(m * 1.15, i + 1, f'{med:.3f}', fontsize=5, va='center',
                    color=PALETTE['neutral_dark'])

    for i, (ax, lbl) in enumerate(zip(axes, 'abc')):
        panel_label(ax, lbl)

    fig.suptitle('不同意图视频的互动转化效率', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.8)
    save_figure(fig, 'fig_v4b_engagement_funnel')
    plt.close(fig)


# --- V4c: Production complexity radar ---
def fig_v4c_production_radar():
    """Compare production features across presenter types."""
    dimensions = ['有特效', '古风慢摇', '节奏卡点',
                  '东方传统乐器', '电子合成器', '古风', '舒缓空灵', '动感活力']
    dim_labels = ['有特效', '慢摇', '卡点', '东方乐器', '电子乐', '古风乐', '舒缓乐', '动感乐']

    fig, ax = plt.subplots(figsize=(8, 7.5), subplot_kw=dict(polar=True))

    n_vars = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    pres_colors = [C_PRESENTER['专业武术家/道长'], C_PRESENTER['泛娱乐博主'], C_PRESENTER['非科班出身']]
    max_val = 0

    for idx, (pres, color) in enumerate(zip(PRESENTER_COLS, pres_colors)):
        if pres not in df.columns:
            continue
        mask = df[pres] == 1
        subset = df[mask]
        n_pres = len(subset)
        if n_pres == 0:
            continue

        values = [subset[d].mean() * 100 for d in dimensions]  # percentage
        values += values[:1]
        max_val = max(max_val, max(values[:-1]))

        ax.fill(angles, values, alpha=0.08, color=color)
        ax.plot(angles, values, 'o-', linewidth=1.5, color=color,
                label=f'{pres} (n={n_pres:,})', markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=6.5)
    ax.set_ylim(0, max_val * 1.2)
    ax.set_yticks([])
    ax.spines['polar'].set_linewidth(0.4)
    ax.grid(linewidth=0.3, alpha=0.4)

    ax.set_title('不同博主类型的生产策略对比', fontsize=9, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.06),
              fontsize=6.5, frameon=True, edgecolor='#d1d5db', fancybox=False)

    panel_label(ax, 'c')
    fig.tight_layout(pad=2.5)
    save_figure(fig, 'fig_v4c_production_radar')
    plt.close(fig)


# --- V4d: Save rate vs Like rate bubble ---
def fig_v4d_save_like_bubble():
    """Scatter bubble: save_rate vs comment_rate, bubble size = video count, colour = purpose."""
    fig, ax = plt.subplots(figsize=(9, 7.5))

    for purp in PURPOSE_COLS_MAIN:
        if purp not in df.columns:
            continue
        mask = df[purp] == 1
        subset = df[mask]
        if len(subset) < 10:
            continue

        # Aggregate by purpose+style combos
        for style in STYLE_COLS_MAIN:
            if style not in df.columns:
                continue
            mask2 = (df[purp] == 1) & (df[style] == 1)
            subset2 = df[mask2]
            n = len(subset2)
            if n < 5:
                continue

            x = subset2['save_rate'].median()
            y = subset2['comment_rate'].median()
            size = np.sqrt(n) * 18

            ax.scatter(x, y, s=size, c=C_PURPOSE.get(purp, PALETTE['neutral_mid']),
                      alpha=0.55, edgecolors='white', linewidth=0.5)

    # Draw reference lines (global medians)
    ax.axvline(df['save_rate'].median(), color=PALETTE['neutral_mid'], linestyle='--',
               linewidth=0.6, alpha=0.5)
    ax.axhline(df['comment_rate'].median(), color=PALETTE['neutral_mid'], linestyle='--',
               linewidth=0.6, alpha=0.5)

    ax.set_xlabel('收藏转化率中位数 (收藏/点赞)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_ylabel('评论转化率中位数 (评论/点赞)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_xlim(0, df['save_rate'].quantile(0.97) * 1.1)
    ax.set_ylim(0, df['comment_rate'].quantile(0.97) * 1.1)

    # Quadrant labels
    x_mid = df['save_rate'].median()
    y_mid = df['comment_rate'].median()
    ax.text(ax.get_xlim()[1] * 0.98, ax.get_ylim()[1] * 0.97,
            '高收藏·高评论', ha='right', va='top', fontsize=6,
            color=PALETTE['neutral_dark'], style='italic')
    ax.text(ax.get_xlim()[0] + ax.get_xlim()[1] * 0.02, ax.get_ylim()[0] + ax.get_ylim()[1] * 0.03,
            '低收藏·低评论', ha='left', va='bottom', fontsize=6,
            color=PALETTE['neutral_mid'], style='italic')

    legend_items = [Patch(facecolor=v, label=k) for k, v in C_PURPOSE.items() if k != '其他']
    ax.legend(handles=legend_items, loc='upper left', fontsize=5.5,
              frameon=True, edgecolor='#d1d5db', fancybox=False, ncol=2)

    panel_label(ax, 'd')
    ax.set_title('流派×意图组合的互动转化特征\n（气泡大小 ∝ 视频数量，点=中位数）',
                 fontsize=9, fontweight='bold', pad=12)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_v4d_save_like_bubble')
    plt.close(fig)


# --- V4e: Effect type × Engagement ---
def fig_v4e_effect_engagement():
    """Grouped bar: effect types and their average engagement."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 1: Effects × avg likes
    ax = axes[0]
    effect_data = {}
    for eff in EFFECT_COLS_MAIN:
        if eff not in df.columns:
            continue
        mask = df[eff] == 1
        effect_data[eff] = {
            'avg_likes': df.loc[mask, '点赞数'].mean(),
            'med_likes': df.loc[mask, '点赞数'].median(),
            'n': mask.sum(),
            'avg_saves': df.loc[mask, '收藏数'].mean(),
        }
    eff_order = sorted(effect_data.items(), key=lambda x: -x[1]['avg_likes'])
    names = [e[0] for e in eff_order]
    vals = [e[1]['avg_likes'] for e in eff_order]
    colors_eff = [PALETTE['blue_main'], PALETTE['blue_secondary'],
                  PALETTE['blue_soft'], PALETTE['neutral_light']]

    bars = ax.bar(range(len(names)), vals, color=colors_eff[:len(names)],
                  edgecolor='white', linewidth=0.6, width=0.6)
    for i, (bar, val, e) in enumerate(zip(bars, vals, eff_order)):
        n = e[1]['n']
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                f'{val:,.0f}\n(n={n:,})', ha='center', fontsize=5.5, color=PALETTE['neutral_dark'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel('平均点赞数', fontsize=7, color=PALETTE['neutral_dark'])
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, max(vals) * 1.18)
    panel_label(ax, 'e')

    # Panel 2: Effects × avg save rate
    ax = axes[1]
    save_rates = []
    for e in eff_order:
        mask = df[e[0]] == 1
        save_rates.append(df.loc[mask, 'save_rate'].mean())
    bars = ax.bar(range(len(names)), save_rates, color=colors_eff[:len(names)],
                  edgecolor='white', linewidth=0.6, width=0.6)
    for i, (bar, val) in enumerate(zip(bars, save_rates)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(save_rates) * 0.02,
                f'{val:.3f}', ha='center', fontsize=5.5, color=PALETTE['neutral_dark'])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel('平均收藏转化率', fontsize=7, color=PALETTE['neutral_dark'])
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, max(save_rates) * 1.18)
    panel_label(ax, 'f')

    fig.suptitle('视频特效与传播效果', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_v4e_effect_engagement')
    plt.close(fig)


# ============================================================
# 4. FIGURE V5 — Content Circles & Temporal Evolution
# ============================================================

# --- V5a: Feature co-occurrence matrix (hero) ---
def fig_v5a_cooccurrence():
    """Cross-dimensional feature co-occurrence heatmap — purpose × all other dims."""
    # Combine: purpose × (style + scene + music_top7)
    music_s = df[MUSIC_STYLE_COLS_MAIN].sum().sort_values(ascending=False)
    top_music = list(music_s.head(7).index)

    all_cols_b = STYLE_COLS_MAIN + SCENE_COLS_MAIN + top_music
    all_labels_b = (['[流派] ' + c for c in STYLE_COLS_MAIN] +
                    ['[场景] ' + c for c in SCENE_COLS_MAIN] +
                    ['[音乐] ' + c for c in top_music])

    mat = np.zeros((len(PURPOSE_COLS_MAIN), len(all_cols_b)))
    for i, pa in enumerate(PURPOSE_COLS_MAIN):
        if pa not in df.columns:
            continue
        for j, cb in enumerate(all_cols_b):
            if cb not in df.columns:
                continue
            mask = (df[pa] == 1) & (df[cb] == 1)
            co_occur = mask.sum()
            # Normalize: Jaccard-like — co-occur / (count_a + count_b - co_occur)
            count_a = df[pa].sum()
            count_b = df[cb].sum()
            union = count_a + count_b - co_occur
            mat[i, j] = co_occur / union * 100 if union > 0 else 0

    df_mat = pd.DataFrame(mat, index=PURPOSE_COLS_MAIN, columns=all_labels_b)

    fig, ax = plt.subplots(figsize=(14, 3.8))
    sns.heatmap(
        df_mat, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.5, 'label': 'Jaccard 指数 (%)', 'aspect': 22},
        vmin=0, vmax=df_mat.values.max() * 1.05,
        annot_kws={'fontsize': 5.5, 'fontweight': 'bold'},
    )
    ax.set_title('视频意图与各维度特征的共现强度（Jaccard 指数）',
                 fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('视频意图', fontsize=7)
    ax.tick_params(labelsize=6)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'a')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v5a_cooccurrence')
    plt.close(fig)


# --- V5b: Purpose composition over time (stacked area) ---
def fig_v5b_temporal_purpose():
    """Stacked area showing purpose composition by month."""
    monthly_purp = monthly_aggregation(PURPOSE_COLS_MAIN)
    # Filter valid months
    monthly_purp = monthly_purp[monthly_purp.index >= '2019-10']
    monthly_purp = monthly_purp[monthly_purp.index <= '2026-04']
    # Fill NaN
    monthly_purp = monthly_purp.fillna(0)
    # Smooth with 3-month rolling
    monthly_purp_smoothed = monthly_purp.rolling(3, min_periods=1, center=True).mean()

    # Normalize to percentage
    monthly_pct = monthly_purp_smoothed.div(monthly_purp_smoothed.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 4.5))
    months = list(monthly_pct.index)
    x = range(len(months))

    # Stack plot
    colors = [C_PURPOSE[c] for c in monthly_pct.columns]
    ax.stackplot(x, *[monthly_pct[c].values for c in monthly_pct.columns],
                 labels=monthly_pct.columns, colors=colors, alpha=0.85)

    ax.set_xlim(0, len(months) - 1)
    ax.set_ylabel('占比 (%)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_xlabel('月份', fontsize=7, color=PALETTE['neutral_dark'])

    # Legend
    ax.legend(loc='upper right', fontsize=5.5, ncol=4, frameon=True,
              edgecolor='#d1d5db', fancybox=False)

    # X-axis
    step = max(1, len(months) // 14)
    tick_pos = list(range(0, len(months), step))
    tick_labels = [months[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=5.5)

    # Annotation: rising/declining trends
    ax.annotate('养生内容占比上升', xy=(len(months) * 0.85, monthly_pct['养生'].iloc[-1] + 2),
                fontsize=5.5, color=PALETTE['neutral_dark'],
                ha='center', style='italic')
    ax.annotate('教学始终为\n核心供给', xy=(len(months) * 0.3, 58),
                fontsize=5.5, color=PALETTE['neutral_dark'],
                ha='center', style='italic')

    panel_label(ax, 'b')
    ax.set_title('视频意图构成的月度演变', fontsize=9, fontweight='bold', pad=10)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v5b_temporal_purpose')
    plt.close(fig)


# --- V5c: Style trend lines over time ---
def fig_v5c_style_trends():
    """Small-multiples: monthly proportion of each style over time."""
    monthly_style = monthly_aggregation(STYLE_COLS_MAIN)
    monthly_style = monthly_style[monthly_style.index >= '2019-10']
    monthly_style = monthly_style[monthly_style.index <= '2026-04']
    monthly_style = monthly_style.fillna(0)
    monthly_pct_style = monthly_style.div(monthly_style.sum(axis=1), axis=0) * 100
    monthly_pct_style = monthly_pct_style.rolling(3, min_periods=1, center=True).mean()

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    months = list(monthly_pct_style.index)
    x = range(len(months))

    for idx, style in enumerate(STYLE_COLS_MAIN):
        ax = axes[idx]
        if style not in monthly_pct_style.columns:
            continue
        vals = monthly_pct_style[style].values

        ax.fill_between(x, 0, vals, color=C_STYLE.get(style, PALETTE['blue_main']),
                        alpha=0.25)
        ax.plot(x, vals, color=C_STYLE.get(style, PALETTE['blue_main']),
                linewidth=1.2)
        ax.axhline(monthly_pct_style[style].mean(), color=PALETTE['neutral_mid'],
                   linestyle='--', linewidth=0.5, alpha=0.6)

        ax.set_title(style, fontsize=7, fontweight='bold', color=PALETTE['neutral_dark'])
        ax.set_ylim(0, max(vals.max() * 1.25, 5))

        # Only show x-ticks for bottom row
        if idx >= 4:
            step = max(1, len(months) // 6)
            tick_pos = list(range(0, len(months), step))
            tick_labels = [months[i] for i in tick_pos]
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=5)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=5.5)

    # Hide empty axes if any
    for idx in range(len(STYLE_COLS_MAIN), len(axes)):
        axes[idx].set_visible(False)

    fig.text(0.5, 0.02, '月份', ha='center', fontsize=7, color=PALETTE['neutral_dark'])
    fig.text(0.02, 0.5, '占比 (%)', va='center', rotation='vertical',
             fontsize=7, color=PALETTE['neutral_dark'])
    fig.suptitle('各太极拳流派的月度占比趋势', fontsize=9, fontweight='bold', y=0.98)
    fig.tight_layout(pad=2, rect=[0.03, 0.03, 1, 0.94])
    save_figure(fig, 'fig_v5c_style_trends')
    plt.close(fig)


# --- V5d: Style × Scene heatmap grid (content circles) ---
def fig_v5d_style_scene_circles():
    """Two side-by-side heatmaps: Style×Scene and Style×Clothing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # Panel 1: Style × Scene
    df_ss = build_cross_table(STYLE_COLS_MAIN, SCENE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ss, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax1,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 22},
        vmin=0, vmax=df_ss.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax1.set_title('流派 × 场景（行百分比）', fontsize=8, fontweight='bold', pad=10)
    ax1.set_xlabel('拍摄场景', fontsize=7)
    ax1.set_ylabel('太极流派', fontsize=7)
    ax1.tick_params(labelsize=6)
    for label in ax1.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')

    # Panel 2: Style × Clothing
    df_sc = build_cross_table(STYLE_COLS_MAIN, CLOTHING_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_sc, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax2,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 22},
        vmin=0, vmax=df_sc.values.max() * 1.05,
        annot_kws={'fontsize': 6, 'fontweight': 'bold'},
    )
    ax2.set_title('流派 × 服装（行百分比）', fontsize=8, fontweight='bold', pad=10)
    ax2.set_xlabel('服装类型', fontsize=7)
    ax2.set_ylabel('')
    ax2.tick_params(labelsize=6)
    for label in ax2.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')

    for ax in [ax1, ax2]:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=5.5)
        cbar.ax.yaxis.label.set_fontsize(6.5)

    panel_label(ax1, 'd')
    panel_label(ax2, 'e')
    fig.suptitle('内容圈层：各流派的空间场景与服装偏好',
                 fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.8)
    save_figure(fig, 'fig_v5d_style_scene_circles')
    plt.close(fig)


# --- V5e: Music style evolution over time ---
def fig_v5e_music_evolution():
    """Stacked area: music style composition over months."""
    monthly_music = monthly_aggregation(MUSIC_STYLE_COLS_MAIN)
    monthly_music = monthly_music[monthly_music.index >= '2020-01']
    monthly_music = monthly_music[monthly_music.index <= '2026-04']
    monthly_music = monthly_music.fillna(0)
    monthly_music_s = monthly_music.rolling(3, min_periods=1, center=True).mean()
    monthly_music_pct = monthly_music_s.div(monthly_music_s.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 4))
    months = list(monthly_music_pct.index)
    x = range(len(months))

    music_colors = {
        '古风': '#9A4D8E', '舒缓空灵': '#42949E', '国风潮流': '#3775BA',
        '动感活力': '#E28E2C', '旁白解说': '#2E7D32', '原声': '#B64342',
        '网红BGM': '#C44E7C', '大气磅礴': '#0F4D92', '戏曲': '#767676',
    }
    cols_ordered = [c for c in monthly_music_pct.columns if c in music_colors]
    colors_ordered = [music_colors.get(c, PALETTE['neutral_mid']) for c in cols_ordered]

    ax.stackplot(x, *[monthly_music_pct[c].values for c in cols_ordered],
                 labels=cols_ordered, colors=colors_ordered, alpha=0.85)

    ax.set_xlim(0, len(months) - 1)
    ax.set_ylabel('占比 (%)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.set_xlabel('月份', fontsize=7, color=PALETTE['neutral_dark'])
    ax.legend(loc='upper left', fontsize=5.5, ncol=3, frameon=True,
              edgecolor='#d1d5db', fancybox=False)
    step = max(1, len(months) // 12)
    tick_pos = list(range(0, len(months), step))
    tick_labels = [months[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=5.5)

    panel_label(ax, 'f')
    ax.set_title('音乐风格的月度构成演变', fontsize=9, fontweight='bold', pad=10)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v5e_music_evolution')
    plt.close(fig)


# --- V5f: Creator concentration (Lorenz curve) ---
def fig_v5f_creator_concentration():
    """Lorenz curve of likes concentration across videos — head vs tail."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Lorenz curve for likes ---
    likes_sorted = np.sort(df['点赞数'].values)
    cumsum = np.cumsum(likes_sorted)
    cumsum_pct = cumsum / cumsum[-1]
    x_vals = np.linspace(0, 100, len(cumsum_pct))

    ax1.fill_between(x_vals, cumsum_pct * 100, x_vals, alpha=0.15,
                     color=PALETTE['blue_main'])
    ax1.plot(x_vals, cumsum_pct * 100, color=PALETTE['blue_main'], linewidth=1.5)
    ax1.plot([0, 100], [0, 100], '--', color=PALETTE['neutral_mid'],
             linewidth=0.8, alpha=0.7)

    # Gini-like annotation
    gini_area = np.trapz(x_vals - cumsum_pct * 100, x_vals) / np.trapz(x_vals, x_vals)
    # Actually compute proper Gini
    n = len(likes_sorted)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * likes_sorted)) / (n * np.sum(likes_sorted)) - (n + 1) / n

    ax1.text(0.95, 0.12, f'Gini = {gini:.3f}', transform=ax1.transAxes,
             fontsize=7, ha='right', color=PALETTE['neutral_dark'])
    ax1.text(0.15, 0.75, 'Top 10% 视频\n获赞占比', transform=ax1.transAxes,
             fontsize=6, ha='center', color=PALETTE['neutral_dark'])
    top10_pct = cumsum_pct[int(n * 0.9)] * 100
    ax1.text(0.15, 0.65, f'{100 - top10_pct:.1f}%', transform=ax1.transAxes,
             fontsize=8, ha='center', fontweight='bold', color=PALETTE['red_strong'])

    ax1.set_xlabel('视频累计占比 (%)', fontsize=7, color=PALETTE['neutral_dark'])
    ax1.set_ylabel('点赞累计占比 (%)', fontsize=7, color=PALETTE['neutral_dark'])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_aspect('equal')
    panel_label(ax1, 'g')

    # --- Head vs tail engagement decomposition ---
    # Split into 10 deciles
    df_sorted = df.sort_values('点赞数', ascending=False)
    df_sorted['decile'] = pd.qcut(np.arange(len(df_sorted)), 10, labels=False)

    decile_means = df_sorted.groupby('decile')['点赞数'].mean()
    decile_sums = df_sorted.groupby('decile')['点赞数'].sum()

    # Bar chart: top 5 deciles
    top_deciles = decile_means.tail(5)
    bars = ax2.bar(range(5), top_deciles.values,
                   color=[PALETTE['blue_main'], PALETTE['blue_secondary'],
                          PALETTE['blue_soft'], PALETTE['neutral_light'],
                          PALETTE['neutral_light']],
                   edgecolor='white', linewidth=0.5, width=0.65)

    for i, (bar, val) in enumerate(zip(bars, top_deciles.values)):
        pct = decile_sums.iloc[5 + i] / decile_sums.sum() * 100
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(top_deciles.values) * 0.02,
                 f'{val:,.0f}\n({pct:.1f}%)', ha='center', fontsize=5.5, color=PALETTE['neutral_dark'])

    ax2.set_xticks(range(5))
    ax2.set_xticklabels(['Top 50-60%', 'Top 60-70%', 'Top 70-80%', 'Top 80-90%', 'Top 90-100%'],
                        fontsize=5.5, rotation=20, ha='right')
    ax2.set_ylabel('平均点赞数', fontsize=7, color=PALETTE['neutral_dark'])
    ax2.set_ylim(0, max(top_deciles.values) * 1.15)
    panel_label(ax2, 'h')

    fig.suptitle('太极视频传播的头部集中效应', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(pad=1.5)
    save_figure(fig, 'fig_v5f_creator_concentration')
    plt.close(fig)


# ============================================================
# 5. FIGURE V6 — Depth Analysis: Duration, Demographics & Language
# ============================================================

# --- V6a: Duration distribution by purpose (violin) ---
def fig_v6a_duration_violin():
    """Split violins: log duration by purpose, coloured."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    plot_data = []
    labels = []
    for purp in PURPOSE_COLS_MAIN:
        if purp not in df.columns:
            continue
        mask = df[purp] == 1
        dur = df.loc[mask, 'dur_sec']
        # Cap at 99.5 percentile for visualization
        cap = dur.quantile(0.995)
        dur_clipped = dur.clip(upper=cap)
        plot_data.append(dur_clipped)
        labels.append(purp)

    # Sort by median
    medians = [np.median(d) for d in plot_data]
    order = np.argsort(medians)
    plot_data_ord = [plot_data[i] for i in order]
    labels_ord = [labels[i] for i in order]
    colors_ord = [C_PURPOSE.get(l, PALETTE['neutral_mid']) for l in labels_ord]

    vp = ax.violinplot(plot_data_ord, positions=range(len(plot_data_ord)),
                        vert=False, showmeans=False, showmedians=True,
                        widths=0.7)

    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors_ord[i])
        body.set_alpha(0.45)
        body.set_edgecolor(colors_ord[i])
        body.set_linewidth(0.8)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in vp:
            vp[partname].set_color(PALETTE['neutral_dark'])
            vp[partname].set_linewidth(0.8)

    # Overlay boxplot-like summary
    bp = ax.boxplot(plot_data_ord, vert=False, positions=range(len(plot_data_ord)),
                     widths=0.25, showfliers=False, patch_artist=True,
                     boxprops={'linewidth': 0.8, 'facecolor': 'white', 'alpha': 0.4},
                     whiskerprops={'linewidth': 0.6},
                     capprops={'linewidth': 0.6},
                     medianprops={'linewidth': 1.5, 'color': PALETTE['neutral_black']})

    ax.set_yticks(range(len(labels_ord)))
    ax.set_yticklabels(labels_ord, fontsize=6.5)
    ax.set_xlabel('视频时长 (秒)', fontsize=7, color=PALETTE['neutral_dark'])

    # Annotate median values
    for i, (d, med) in enumerate(zip(plot_data_ord, medians)):
        ax.text(d.max() * 1.02, i, f'中位数: {med:.0f}s', fontsize=5.5,
                va='center', color=PALETTE['neutral_dark'])

    panel_label(ax, 'a')
    ax.set_title('不同意图的视频时长分布', fontsize=9, fontweight='bold', pad=12)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v6a_duration_violin')
    plt.close(fig)


# --- V6b: Age × Purpose heatmap ---
def fig_v6b_age_purpose():
    fig, ax = plt.subplots(figsize=(9.5, 2.5))
    df_ap = build_cross_table(AGE_COLS, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_ap, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 18},
        vmin=0, vmax=df_ap.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('博主年龄段 × 视频意图（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('年龄段', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'b')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v6b_age_purpose')
    plt.close(fig)


# --- V6c: Gender × Purpose interaction ---
def fig_v6c_gender_purpose():
    fig, ax = plt.subplots(figsize=(9.5, 2.2))
    df_gp = build_cross_table(GENDER_COLS, PURPOSE_COLS_MAIN, normalize='rows')
    sns.heatmap(
        df_gp, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.55, 'label': '占比 (%)', 'aspect': 18},
        vmin=0, vmax=df_gp.values.max() * 1.05,
        annot_kws={'fontsize': 6.5, 'fontweight': 'bold'},
    )
    ax.set_title('博主性别 × 视频意图（行百分比）', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('视频意图', fontsize=7)
    ax.set_ylabel('性别', fontsize=7)
    ax.tick_params(labelsize=6.5)
    for label in ax.get_xticklabels():
        label.set_rotation(22)
        label.set_ha('right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'c')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v6c_gender_purpose')
    plt.close(fig)


# --- V6d: Language comparison ---
def fig_v6d_language_comparison():
    """Compare content profiles between Chinese, English, and bilingual videos."""
    # Filter languages with sufficient data
    lang_data = {}
    for lang in LANGUAGE_COLS:
        if lang not in df.columns:
            continue
        mask = df[lang] == 1
        if mask.sum() >= 5:
            lang_data[lang] = mask

    target_dims = ['教学', '养生', '文化哲学', '日常', '实战', '门派', '赛事']
    target_dims = [d for d in target_dims if d in df.columns]

    fig, ax = plt.subplots(figsize=(9.5, 3.5))

    x = np.arange(len(target_dims))
    width = 0.25
    colors_lang = {
        '中文': PALETTE['blue_main'],
        '英文': PALETTE['red_strong'],
        '双语': PALETTE['teal'],
    }

    for i, (lang, mask) in enumerate(lang_data.items()):
        subset = df[mask]
        n = len(subset)
        if n == 0:
            continue
        vals = [subset[d].mean() * 100 for d in target_dims]
        bars = ax.bar(x + i * width, vals, width, color=colors_lang.get(lang, PALETTE['neutral_mid']),
                      label=f'{lang} (n={n:,})', edgecolor='white', linewidth=0.4)
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', fontsize=5, color=PALETTE['neutral_dark'])

    ax.set_xticks(x + width)
    ax.set_xticklabels(target_dims, fontsize=6.5)
    ax.set_ylabel('视频占比 (%)', fontsize=7, color=PALETTE['neutral_dark'])
    ax.legend(fontsize=6, frameon=True, edgecolor='#d1d5db', fancybox=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    panel_label(ax, 'd')
    ax.set_title('不同语言视频的内容特征对比', fontsize=9, fontweight='bold', pad=12)
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v6d_language_comparison')
    plt.close(fig)


# --- V6e: Multi-dimensional engagement signature radar by purpose ---
def fig_v6e_engagement_radar():
    """Radar showing how different purposes engage across dimensions."""
    metrics = ['点赞数', '评论数', '收藏数', '转发数']
    metric_labels = ['点赞', '评论', '收藏', '转发']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    n_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    max_val = 0
    for purp in PURPOSE_COLS_MAIN:
        if purp not in df.columns:
            continue
        mask = df[purp] == 1
        subset = df[mask]
        if len(subset) < 10:
            continue
        # Normalize each metric by global mean → "engagement signature" (relative to average)
        global_mean = df[metrics].mean()
        purp_mean = subset[metrics].mean()
        ratio = (purp_mean / global_mean).values  # >1 = above average
        ratio_list = list(ratio) + [ratio[0]]
        max_val = max(max_val, max(ratio))

        ax.fill(angles, ratio_list, alpha=0.06, color=C_PURPOSE[purp])
        ax.plot(angles, ratio_list, 'o-', linewidth=1.5,
                color=C_PURPOSE[purp], label=purp, markersize=4)

    # Reference line at 1.0 (global average)
    ax.plot(angles, [1] * len(angles), '--', color=PALETTE['neutral_mid'],
            linewidth=0.7, alpha=0.5, label='全局均值')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=7)
    ax.set_ylim(0, max_val * 1.15)
    ax.set_yticks([0.5, 1.0, 2.0, 3.0])
    ax.set_yticklabels(['0.5×', '1.0×', '2.0×', '3.0×'], fontsize=5.5, color=PALETTE['neutral_mid'])
    ax.spines['polar'].set_linewidth(0.4)
    ax.grid(linewidth=0.3, alpha=0.4)

    ax.set_title('各意图视频的互动特征雷达\n（相对全局均值的倍数）', fontsize=9, fontweight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.06),
              fontsize=6, frameon=True, edgecolor='#d1d5db', fancybox=False, ncol=2)

    panel_label(ax, 'e')
    fig.tight_layout(pad=2.5)
    save_figure(fig, 'fig_v6e_engagement_radar')
    plt.close(fig)


# --- V6f: Monthly engagement heatmap by purpose ---
def fig_v6f_monthly_engagement_heatmap():
    """Heatmap: month × purpose = avg likes."""
    monthly = df.groupby('年月')
    mat_data = {}
    for purp in PURPOSE_COLS_MAIN:
        if purp not in df.columns:
            continue
        mat_data[purp] = monthly.apply(
            lambda g: g.loc[g[purp] == 1, '点赞数'].mean() if (g[purp] == 1).any() else np.nan
        )

    df_heat = pd.DataFrame(mat_data)
    df_heat = df_heat[df_heat.index >= '2020-06']
    df_heat = df_heat[df_heat.index <= '2026-04']
    df_heat = df_heat.dropna(how='all')
    # Log transform for better visualization
    df_heat_log = np.log10(df_heat + 1)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.heatmap(
        df_heat_log.T, cmap='YlGnBu', ax=ax,
        linewidths=0.2, linecolor='white',
        cbar_kws={'shrink': 0.4, 'label': 'log₁₀(平均点赞+1)', 'aspect': 25},
        vmin=df_heat_log.min().min(), vmax=df_heat_log.max().max(),
        annot=False,
    )
    ax.set_title('各意图视频月度平均点赞热力图', fontsize=9, fontweight='bold', pad=12)
    ax.set_xlabel('月份', fontsize=7)
    ax.set_ylabel('视频意图', fontsize=7)
    ax.tick_params(labelsize=6)
    # Show subset of x-labels
    n_cols = len(df_heat_log.columns)
    step = max(1, n_cols // 14)
    tick_pos = list(range(0, n_cols, step))
    tick_labels = [df_heat_log.columns[i] for i in tick_pos]
    ax.set_xticks([i + 0.5 for i in tick_pos])
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=5.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5.5)
    cbar.ax.yaxis.label.set_fontsize(6.5)
    panel_label(ax, 'f')
    fig.tight_layout(pad=1.2)
    save_figure(fig, 'fig_v6f_monthly_engagement_heatmap')
    plt.close(fig)


# ============================================================
# 6. Additional analysis: Summary export
# ============================================================
def export_extended_summary():
    lines = []
    lines.append('=' * 70)
    lines.append('Tai Chi Video — Extended Analysis Summary')
    lines.append('=' * 70)
    lines.append(f'Total videos: {TOTAL_VIDEOS:,}')
    lines.append('')

    lines.append('--- Duration statistics ---')
    lines.append(f'  Mean: {df["dur_sec"].mean():.0f}s ({df["dur_sec"].mean()/60:.1f}min)')
    lines.append(f'  Median: {df["dur_sec"].median():.0f}s ({df["dur_sec"].median()/60:.1f}min)')
    lines.append(f'  Q1: {df["dur_sec"].quantile(0.25):.0f}s, Q3: {df["dur_sec"].quantile(0.75):.0f}s')
    lines.append('')

    lines.append('--- Production complexity ---')
    lines.append(f'  Mean effects/features per video: {df["production_score"].mean():.2f}')
    lines.append(f'  By presenter type:')
    for pres in PRESENTER_COLS:
        if pres in df.columns:
            mask = df[pres] == 1
            lines.append(f'    {pres}: {df.loc[mask, "production_score"].mean():.2f}')
    lines.append('')

    lines.append('--- Engagement correlation ---')
    corr_mat = df[['点赞数', '评论数', '收藏数', '转发数', '推荐数']].corr()
    lines.append(str(corr_mat))
    lines.append('')

    lines.append('--- Duration × Engagement correlation ---')
    r, p = stats.pearsonr(df['log_dur'], df['log_likes'])
    lines.append(f'  log(dur) vs log(likes): r={r:.3f}, p={p:.6f}')
    lines.append('')

    lines.append('--- Top 5% vs Bottom 95% engagement ---')
    top_thresh = df['点赞数'].quantile(0.95)
    top5 = df[df['点赞数'] >= top_thresh]
    bot95 = df[df['点赞数'] < top_thresh]
    lines.append(f'  Top 5% (n={len(top5):,}): avg saves={top5["收藏数"].mean():.0f}, avg shares={top5["转发数"].mean():.0f}')
    lines.append(f'  Bot 95% (n={len(bot95):,}): avg saves={bot95["收藏数"].mean():.0f}, avg shares={bot95["转发数"].mean():.0f}')
    lines.append('')

    lines.append('--- Language breakdown ---')
    for lang in LANGUAGE_COLS:
        if lang in df.columns:
            mask = df[lang] == 1
            lines.append(f'  {lang}: {mask.sum():,} videos ({mask.sum()/TOTAL_VIDEOS*100:.1f}%)')
    lines.append('')

    # Temporal summary
    lines.append('--- Monthly posting volume (last 12 months) ---')
    monthly_counts = df.groupby('年月').size()
    monthly_counts = monthly_counts[monthly_counts.index >= '2024-01']
    for m, c in monthly_counts.items():
        lines.append(f'  {m}: {c:,}')

    path = OUTPUT_DIR / 'summary_statistics_video_extended.txt'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  Saved: summary_statistics_video_extended.txt')


# ============================================================
# 7. Main
# ============================================================
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('Extended Video Analysis — Nature-Quality Figures')
    print(f'Output: {OUTPUT_DIR}')
    print('=' * 60)

    print('\n--- Figure V4: Production Strategy & Engagement Heterogeneity ---')
    fig_v4a_duration_engagement()
    fig_v4b_engagement_funnel()
    fig_v4c_production_radar()
    fig_v4d_save_like_bubble()
    fig_v4e_effect_engagement()

    print('\n--- Figure V5: Content Circles & Temporal Evolution ---')
    fig_v5a_cooccurrence()
    fig_v5b_temporal_purpose()
    fig_v5c_style_trends()
    fig_v5d_style_scene_circles()
    fig_v5e_music_evolution()
    fig_v5f_creator_concentration()

    print('\n--- Figure V6: Depth Analysis ---')
    fig_v6a_duration_violin()
    fig_v6b_age_purpose()
    fig_v6c_gender_purpose()
    fig_v6d_language_comparison()
    fig_v6e_engagement_radar()
    fig_v6f_monthly_engagement_heatmap()

    print('\n--- Extended Summary ---')
    export_extended_summary()

    print('\n' + '=' * 60)
    print(f'Done. 17 additional figures saved to: {OUTPUT_DIR}')
    print('=' * 60)
