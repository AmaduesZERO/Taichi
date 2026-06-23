# TaiChi — 太极拳短视频文化传承认知分析

> Tai Chi Cultural Heritage Cognitive Analysis on Douyin (TikTok)



```
TaiChi/
├── src/                            # 源码
│   ├── data/                       # 数据处理管线
│   │   ├── processing.py           # 合并/筛选/去重/清洗/分割
│   │   ├── clean_ip.py             # IP属地清洗（仅保留中国省份）
│   │   ├── onehot_ip.py            # IP属地独热编码
│   │   ├── verify_merge.py         # 合并数据完整性验证
│   │   └── inspect.py              # 数据文件信息查看
│   │
│   ├── labeling/                   # LLM标注
│   │   ├── xiaomi_label.py         # 小米Mimo模型视频标签识别
│   │   ├── douyin_label.py         # 豆包大模型视频标签识别
│   │   └── classify_comments.py    # DeepSeek评论多标签分类
│   │
│   ├── analysis/                   # 分析 & 制图
│   │   ├── comment_cognition.py    # 评论认知分析 
│   │   ├── video_content.py        # 视频内容分析 
│   │   ├── video_extended.py       # 视频扩展分析
│   │   ├── fusion.py              # 视频-评论融合分析
│   │   └── china_heatmap.py        # 中国IP属地热力地图
│   │
│   ├── media/                      # 媒体处理
│   │   ├── video_to_base64.py      # 视频转Base64编码
│   │   └── trim_videos.py          # 视频裁剪（ffmpeg）
│   │
│   └── utils/                      # 共享工具
│       ├── config.py               # 统一路径配置（所有路径集中管理）
│       └── references/             # 列名参考文件
│           ├── video_cols.txt      # 视频数据列名
│           └── comment_cols.txt    # 评论数据列名
│
├── dy_file/                        # 数据目录
│   ├── dy_video/                   # 原始下载视频 (.mp4)
│   ├── dy_video_base64/            # Base64编码视频 (.txt)
│   ├── dy_video_csv/               # 处理后CSV + 视频级CSV
│   ├── dy_vloger_comment/          # 原始评论CSV
│   ├── split_all_data/             # (中间产物) 分片视频数据
│   ├── split_all_comments/         # (中间产物) 分片评论数据
│   └── split_all_comments_labeled/ # (中间产物) 已标注分片评论
│
├── output/                         # 图表输出
│   ├── figures_comment/            # 评论分析图 (SVG+PDF+TIFF)
│   ├── figures_video/              # 视频分析图
│   ├── figures_fusion/             # 融合分析图
│   └── heatmap/                    # 热力地图HTML
│
├── .env.example                    # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md
```



```
data/raw/videos/  +  data/raw/comments/
        │
        ▼
  src/data/processing.py (合并/筛选/清洗)
        │
        ▼
  src/labeling/ (LLM标注 — 视频标签 + 评论标签)
        │
        ▼
  src/analysis/ (分析 + 制图)
        │
        ▼
  output/figures_*/ (SVG + PDF + TIFF)
```

## 使用方式

### 环境配置
```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Keys
cp .env.example .env
# 编辑 .env 填入实际密钥
```

### 运行脚本
所有脚本从项目根目录运行：

```bash
# 数据处理
python -m src.data.processing
python -m src.data.clean_ip
python -m src.data.onehot_ip

# 视频标注（需先 video_to_base64）
python -m src.media.video_to_base64
python -m src.labeling.xiaomi_label

# 评论标注
python -m src.labeling.classify_comments

# 分析与制图
python -m src.analysis.comment_cognition
python -m src.analysis.video_content
python -m src.analysis.fusion
```



## 注意事项

- **API密钥**: 所有API密钥通过 `.env` 环境变量管理，不要硬编码到脚本中
- **中间产物**: `split_all_*` 和 `dy_video_base64/` 为管线中间产物，已加入 `.gitignore`
- **输出图表**: `output/` 目录内容为可复现产物，已加入 `.gitignore`
- **路径配置**: 所有输入输出路径统一在 `src/utils/config.py` 中管理
