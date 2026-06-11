import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import COMMENT_LABELED_CLEANED_ONEHOT, HEATMAP_DIR

import pandas as pd
import json
import re
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode

# ============================================================
# 1. Load data and aggregate IP location counts
# ============================================================
input_path = str(COMMENT_LABELED_CLEANED_ONEHOT)
output_html = str(HEATMAP_DIR / 'china_ip_heatmap.html')

print('Loading data...')
df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
ip_cols = [c for c in df.columns if c.startswith('IP属地_') and c != 'IP属地_IP未知']

counts_short = {}
for col in ip_cols:
    province_label = col.replace('IP属地_', '')
    counts_short[province_label] = int(df[col].sum())

print(f'Before mapping — unique locations: {len(counts_short)}')

# ============================================================
# 2. Map short names -> full ECharts administrative names
# ============================================================
municipality_map = {
    '北京': '北京市', '天津': '天津市', '上海': '上海市', '重庆': '重庆市',
}
autonomous_region_map = {
    '广西': '广西壮族自治区', '新疆': '新疆维吾尔自治区',
    '宁夏': '宁夏回族自治区', '西藏': '西藏自治区', '内蒙古': '内蒙古自治区',
}
sar_map = {
    '中国香港': '香港特别行政区', '中国澳门': '澳门特别行政区', '中国台湾': '台湾省',
}
skip_map = {'中国': None, 'IP未知': None}

mapped_data = {}
skipped = []

for short_name, cnt in counts_short.items():
    if short_name in skip_map:
        result = skip_map[short_name]
        if result is None:
            skipped.append((short_name, cnt))
            continue
        mapped_data[result] = mapped_data.get(result, 0) + cnt
    elif short_name in municipality_map:
        mapped_name = municipality_map[short_name]
        mapped_data[mapped_name] = mapped_data.get(mapped_name, 0) + cnt
    elif short_name in autonomous_region_map:
        mapped_name = autonomous_region_map[short_name]
        mapped_data[mapped_name] = mapped_data.get(mapped_name, 0) + cnt
    elif short_name in sar_map:
        mapped_name = sar_map[short_name]
        mapped_data[mapped_name] = mapped_data.get(mapped_name, 0) + cnt
    else:
        mapped_name = short_name + '省'
        mapped_data[mapped_name] = mapped_data.get(mapped_name, 0) + cnt

print(f'Mapped to full names: {len(mapped_data)} regions')
print(f'Skipped: {skipped}')

# ============================================================
# 3. Data summary
# ============================================================
data_pairs = [[name, cnt] for name, cnt in mapped_data.items()]
total_sum = sum(mapped_data.values())
max_val = max(mapped_data.values())
min_val = min(mapped_data.values())
print(f'Total comments (excl. IP未知): {total_sum:,}')
print(f'Value range: {min_val:,} ~ {max_val:,}')

# ============================================================
# 4. English name mapping + top3/bottom3
# ============================================================
province_en_map = {
    '北京市': 'Beijing', '天津市': 'Tianjin', '上海市': 'Shanghai', '重庆市': 'Chongqing',
    '河北省': 'Hebei', '山西省': 'Shanxi', '辽宁省': 'Liaoning', '吉林省': 'Jilin',
    '黑龙江省': 'Heilongjiang', '江苏省': 'Jiangsu', '浙江省': 'Zhejiang',
    '安徽省': 'Anhui', '福建省': 'Fujian', '江西省': 'Jiangxi', '山东省': 'Shandong',
    '河南省': 'Henan', '湖北省': 'Hubei', '湖南省': 'Hunan', '广东省': 'Guangdong',
    '海南省': 'Hainan', '四川省': 'Sichuan', '贵州省': 'Guizhou', '云南省': 'Yunnan',
    '陕西省': 'Shaanxi', '甘肃省': 'Gansu', '青海省': 'Qinghai',
    '广西壮族自治区': 'Guangxi', '新疆维吾尔自治区': 'Xinjiang',
    '宁夏回族自治区': 'Ningxia', '西藏自治区': 'Tibet', '内蒙古自治区': 'Inner Mongolia',
    '香港特别行政区': 'Hong Kong', '澳门特别行政区': 'Macau', '台湾省': 'Taiwan',
}

sorted_data = sorted(mapped_data.items(), key=lambda x: -x[1])
top3_ch = [name for name, _ in sorted_data[:3]]
bottom3_ch = [name for name, _ in sorted_data[-3:]]

label_en_list = []
for ch in top3_ch:
    label_en_list.append([ch, province_en_map.get(ch, ch)])
for ch in bottom3_ch:
    label_en_list.append([ch, province_en_map.get(ch, ch)])

label_list_json = json.dumps(label_en_list, ensure_ascii=False)

print(f'Label provinces (Top3 + Bottom3):')
for ch, en in label_en_list:
    print(f'  {ch} -> {en} ({mapped_data[ch]:,})')

# Data panel rows
data_rows_html = ''
for ch_name, cnt in sorted_data:
    en_name = province_en_map.get(ch_name, ch_name)
    data_rows_html += f'<div class="data-row"><span class="data-name">{en_name}</span><span class="data-count">{cnt:,}</span></div>\n'

# ============================================================
# 5. Create heat map (labels OFF for all provinces)
# ============================================================
gradient_colors = [
    '#0b1a3b', '#0c4a6e', '#0e7490', '#0891b2', '#06b6d4',
    '#22d3ee', '#facc15', '#f97316', '#ef4444', '#b91c1c',
]

tooltip_formatter = JsCode("""
    function(params) {
        if (params.data && params.data.value != null) {
            return '<div style="font-family:Microsoft YaHei,sans-serif;font-size:15px;padding:8px 14px;">'
                + '<span style="font-weight:bold;font-size:18px;color:#f8fafc;">' + params.name + '</span><br/>'
                + '<span style="color:#94a3b8;">Comments: </span>'
                + '<span style="font-weight:bold;font-size:20px;color:#facc15;">' 
                + params.data.value.toLocaleString() + '</span>'
                + '</div>';
        }
        return '<div style="font-family:Microsoft YaHei,sans-serif;font-size:14px;padding:6px 12px;color:#94a3b8;">'
            + params.name + '<br/>No data</div>';
    }
""")

map_chart = (
    Map(init_opts=opts.InitOpts(
        width='1600px',
        height='1200px',
        bg_color='#ffffff',
        theme=ThemeType.LIGHT,
        renderer='canvas',
    ))
    .add(
        series_name='Comments',
        data_pair=data_pairs,
        maptype='china',
        is_map_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),  # ALL labels off initially
        itemstyle_opts=opts.ItemStyleOpts(
            border_color='rgba(100,116,139,0.25)',
            border_width=1.5,
            area_color='#f1f5f9',
        ),
        emphasis_itemstyle_opts=opts.ItemStyleOpts(
            border_color='#1e293b',
            border_width=3,
            area_color=None,
        ),
        emphasis_label_opts=opts.LabelOpts(
            is_show=True,
            font_size=24,
            font_weight='bold',
            color='#1e293b',
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger='item',
            formatter=tooltip_formatter,
            background_color='rgba(15,23,42,0.95)',
            border_color='rgba(148,163,184,0.3)',
            border_width=1,
            textstyle_opts=opts.TextStyleOpts(color='#e2e8f0'),
            padding=[10, 16],
        ),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            is_show=True,
            type_='color',
            min_=min_val,
            max_=max_val,
            range_color=gradient_colors,
            pos_left='35px',
            pos_top='center',
            orient='vertical',
            range_text=['High', 'Low'],
            textstyle_opts=opts.TextStyleOpts(
                color='#334155',
                font_size=20,
                font_weight='bold',
                font_family='Microsoft YaHei, sans-serif',
            ),
            item_width=24,
            item_height=200,
            border_color='rgba(100,116,139,0.2)',
            background_color='rgba(248,250,252,0.8)',
            is_calculable=True,
            is_piecewise=False,
        ),
        legend_opts=opts.LegendOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(trigger='item'),
    )
)

# ============================================================
# 6. Render to HTML
# ============================================================
print(f'\nRendering map to {output_html}...')
map_chart.render(output_html)
print('Done!')

# ============================================================
# 7. Post-process: inject labels + data panel
# ============================================================
with open(output_html, 'r', encoding='utf-8') as f:
    html_content = f.read()

# Extract chart info
chart_var_match = re.search(r"var (chart_\w+) = echarts\.init", html_content)
chart_var = chart_var_match.group(1) if chart_var_match else 'chart'
option_var_match = re.search(r"var (option_\w+) = \{", html_content)
option_var = option_var_match.group(1) if option_var_match else 'option'
chart_dom_id_match = re.search(r"document\.getElementById\('([^']+)'\)", html_content)
chart_dom_id = chart_dom_id_match.group(1) if chart_dom_id_match else 'chart'

# JS script that runs AFTER setOption to add labels for top3+bottom3 only
# IMPORTANT: This must be a SEPARATE <script> block OUTSIDE pyecharts' script block
# to avoid HTML parser prematurely closing the outer script.
label_script = f'''
<script>
(function() {{
    var chartDom = document.getElementById('{chart_dom_id}');
    if (!chartDom) return;
    var labelData = {label_list_json};
    var labelMap = {{}};
    for (var i = 0; i < labelData.length; i++) {{
        labelMap[labelData[i][0]] = labelData[i][1];
    }}

    // Wait for pyecharts' script to finish (it's earlier in DOM, so DOMContentLoaded is safe)
    var tryInject = function () {{
        var instance = echarts.getInstanceByDom(chartDom);
        if (!instance) {{
            setTimeout(tryInject, 100);
            return;
        }}
        var opt = instance.getOption();
        if (opt && opt.series && opt.series[0]) {{
            opt.series[0].label = {{
                show: true,
                position: 'inside',
                fontSize: 28,
                fontFamily: 'Segoe UI, Microsoft YaHei, sans-serif',
                color: '#ffffff',
                fontWeight: 'bold',
                textShadowColor: 'rgba(0,0,0,0.7)',
                textShadowBlur: 4,
                formatter: function(params) {{
                    return labelMap[params.name] || '';
                }}
            }};
            instance.setOption(opt);
        }}
    }};
    // Wait for DOMContentLoaded + small delay for echarts.init + setOption
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', function() {{ setTimeout(tryInject, 500); }});
    }} else {{
        setTimeout(tryInject, 500);
    }}
}})();
</script>
'''

# Wrap body content in a relative container first (for panel absolute positioning)
html_content = re.sub(r'<body\s*>', '<body>\n<div style="position:relative;width:100%;height:100%;">', html_content)

# Data panel HTML
data_panel_html = f'''<div id="data-panel" style="
    position: absolute;
    right: 60px;
    top: 80px;
    bottom: 80px;
    width: 340px;
    background: rgba(248, 250, 252, 0.95);
    border: 1.5px solid rgba(100, 116, 139, 0.3);
    border-radius: 14px;
    box-shadow: 0 6px 30px rgba(0,0,0,0.1);
    overflow-y: auto;
    padding: 28px 24px;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    z-index: 100;
">
<div style="
    font-size: 21px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 2.5px solid rgba(100,116,139,0.25);
    letter-spacing: 0.5px;
">Comment Count by Province</div>
<div style="font-size: 15px; color: #64748b; margin-bottom: 12px;">
    Total: {total_sum:,}  ·  {len(sorted_data)} regions
</div>
<div style="display: flex; flex-direction: column; gap: 3px;">
{data_rows_html}
</div>
</div>

<style>
.data-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 14px;
    border-radius: 8px;
    transition: background 0.15s;
}}
.data-row:hover {{ background: rgba(100, 116, 139, 0.1); }}
.data-name {{
    font-size: 15px;
    color: #1e293b;
    font-weight: 600;
}}
.data-count {{
    font-size: 15px;
    color: #0f172a;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}
.data-row:nth-child(1), .data-row:nth-child(2), .data-row:nth-child(3) {{
    background: rgba(245, 158, 11, 0.08);
}}
.data-row:nth-child(1) .data-name,
.data-row:nth-child(2) .data-name,
.data-row:nth-child(3) .data-name {{
    color: #b45309;
}}
.data-row:nth-last-child(1),
.data-row:nth-last-child(2),
.data-row:nth-last-child(3) {{
    background: rgba(100, 116, 139, 0.05);
}}
</style>'''

# Inject data panel BEFORE </div> (closing wrapper) and </body>
html_content = html_content.replace('</body>', f'{data_panel_html}\n</div>\n</body>')

# Then inject label script as separate <script> block OUTSIDE pyecharts' script block
html_content = html_content.replace('</body>', label_script + '\n</body>')

with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_content)

print('Labels and data panel injected successfully!')
print(f'\n=== Summary ===')
print(f'Regions displayed: {len(mapped_data)}')
print(f'Total comments: {total_sum:,}')
print(f'Max: {max(mapped_data.items(), key=lambda x: x[1])[0]} ({max_val:,})')
print(f'Min: {min(mapped_data.items(), key=lambda x: x[1])[0]} ({min_val:,})')