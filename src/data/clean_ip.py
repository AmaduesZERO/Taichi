import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import COMMENT_LABELED, COMMENT_LABELED_CLEANED

import pandas as pd

input_path = str(COMMENT_LABELED)
output_path = str(COMMENT_LABELED_CLEANED)

print('Reading file...')
df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
print(f'Total rows: {len(df)}')

# All Chinese province-level locations + special regions
cn_locations = {
    '北京','天津','上海','重庆',
    '河北','山西','辽宁','吉林','黑龙江',
    '江苏','浙江','安徽','福建','江西','山东',
    '河南','湖北','湖南','广东','广西','海南',
    '四川','贵州','云南','西藏',
    '陕西','甘肃','青海','宁夏','新疆','内蒙古',
    '中国香港','中国澳门','中国台湾','中国',
    'IP未知'
}

# Check all unique values
all_locs = df['评论IP属地'].value_counts()
print(f'Unique IP locations: {len(all_locs)}')

# Show non-Chinese locations that will be removed
non_cn_mask = ~df['评论IP属地'].isin(cn_locations)
non_cn_locs = df.loc[non_cn_mask, '评论IP属地'].value_counts()
print(f'\nNon-Chinese locations to remove ({len(non_cn_locs)} unique, {non_cn_mask.sum()} rows):')
print(non_cn_locs.to_string())

# Keep only Chinese locations
df_clean = df[df['评论IP属地'].isin(cn_locations)]
print(f'\nAfter cleaning: {len(df_clean)} rows')
print(f'Removed: {len(df) - len(df_clean)} rows')

# Save
print(f'\nSaving to {output_path}...')
df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
print('Done!')

# Verify
print(f'\n=== Verification ===')
print(f'Remaining unique locations: {df_clean["评论IP属地"].nunique()}')
print(df_clean['评论IP属地'].value_counts().to_string())