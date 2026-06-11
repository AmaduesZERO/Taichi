import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import DATA_DIR

import pandas as pd
import os, glob
import numpy as np

base = str(DATA_DIR)
labeled_dir = os.path.join(base, 'split_all_comments_labeled')
merged_path = os.path.join(base, 'dy_video_csv', 'merged_dy_data_Allcomments_labeled.csv')

labeled_files = sorted(glob.glob(os.path.join(labeled_dir, 'merged_dy_data_Allcomments_part*_labeled.csv')))

print('Reading merged file...')
merged = pd.read_csv(merged_path, encoding='utf-8-sig', low_memory=False)
print(f'Merged rows: {len(merged)}')

sample_indices = [0, 1, 2, 10, 50, 100, 500, 1000, 1500, 1652]
label_cols_all = [c for c in merged.columns if c.startswith(('情感_','属性一_','属性二_','意向_'))]
comment_col = '评论内容'

# Calculate offsets using actual pandas row counts
offset = 0
file_row_ranges = {}
for idx, lf in enumerate(labeled_files):
    df = pd.read_csv(lf, encoding='utf-8-sig')
    n = len(df)
    file_row_ranges[idx] = (offset, offset + n)
    offset += n

print(f'Total rows from splits (pandas): {offset}')
print(f'Merged rows: {len(merged)}')
print(f'Row count match: {offset == len(merged)}')
print()

all_pass = True
for idx in sample_indices:
    if idx >= len(labeled_files):
        continue
    
    lf = labeled_files[idx]
    split = pd.read_csv(lf, encoding='utf-8-sig')
    n = len(split)
    start = file_row_ranges[idx][0]
    end = file_row_ranges[idx][1]
    
    merged_slice = merged.iloc[start:end]
    
    # Compare comments
    s_com = split[comment_col].fillna('').astype(str).values
    m_com = merged_slice[comment_col].fillna('').astype(str).values
    com_ok = np.array_equal(s_com, m_com)
    
    # Compare labels  
    s_lab = split[label_cols_all].values
    m_lab = merged_slice[label_cols_all].values
    lab_ok = np.array_equal(s_lab, m_lab)
    lab_matches = (s_lab == m_lab).sum()
    lab_total = s_lab.size
    
    status = 'PASS' if (com_ok and lab_ok) else 'FAIL'
    print(f'[{status}] part{idx+1:04d}: rows={n:5d}, offset=[{start:7d},{end:7d})  comments={com_ok}  labels={lab_ok} ({lab_matches}/{lab_total})')

    if not (com_ok and lab_ok):
        all_pass = False
        if not com_ok:
            for i in range(min(n, 5)):
                if s_com[i] != m_com[i]:
                    print(f'    Comment mismatch at row {i}:')
                    sc = str(s_com[i])[:80]
                    mc = str(m_com[i])[:80]
                    print(f'      split:  {sc}')
                    print(f'      merged: {mc}')
                    break
        if not lab_ok:
            mask = ~(s_lab == m_lab)
            mismatch_rows = np.unique(np.where(mask)[0])
            print(f'    Label mismatches in {len(mismatch_rows)} rows (first 5): {mismatch_rows[:5].tolist()}')
            first = mismatch_rows[0]
            s_row = s_lab[first].tolist()
            m_row = m_lab[first].tolist()
            for ci, (sv, mv) in enumerate(zip(s_row, m_row)):
                if sv != mv:
                    col_name = label_cols_all[ci]
                    print(f'      col={col_name}: split={int(sv)}, merged={int(mv)}')

print()
print(f'Overall: {"ALL PASS" if all_pass else "SOME FAILURES"}')