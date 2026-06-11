import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import COMMENT_LABELED_CLEANED, COMMENT_LABELED_CLEANED_ONEHOT

import pandas as pd

input_path = str(COMMENT_LABELED_CLEANED)
output_path = str(COMMENT_LABELED_CLEANED_ONEHOT)

print('Reading file...')
df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')

# Apply one-hot encoding to 评论IP属地
col = '评论IP属地'
print(f'\nUnique values in "{col}": {df[col].nunique()}')
print(df[col].value_counts().to_string())

# Create one-hot encoded columns
dummies = pd.get_dummies(df[col], prefix='IP属地', dtype=int)
print(f'\nOne-hot columns created: {len(dummies.columns)}')

# Drop original column and append one-hot columns
df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
print(f'New shape: {df.shape}')
print(f'One-hot columns: {list(dummies.columns)}')

# Save
print(f'\nSaving to {output_path}...')
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print('Done!')

# Verify
print(f'\n=== Verification ===')
ip_cols = [c for c in df.columns if c.startswith('IP属地_')]
print(f'IP one-hot columns: {len(ip_cols)}')
# Check row sum = 1 for all rows
row_sums = df[ip_cols].sum(axis=1)
print(f'All rows sum to 1: {(row_sums == 1).all()}')
# Check total counts match original
for c in ip_cols:
    total = df[c].sum()
    location = c.replace('IP属地_', '')
    print(f'  {location}: {int(total)}')