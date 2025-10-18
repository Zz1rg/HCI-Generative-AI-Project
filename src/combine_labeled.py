#!/usr/bin/env python
import pandas as pd
from pathlib import Path
import os

curr_dir = Path(__file__).parent.resolve()
src_dir = '../Data/labeled_twitch_gemini-2.0-flash_new_data_w_50ex/'
dst_file = curr_dir / src_dir / 'combined.csv'

content = [
    'message,label',
]

for entry in (curr_dir / src_dir).iterdir():
    if os.path.basename(entry).startswith('output'):
        with open(entry, 'r') as f:
            content.extend(f.read().splitlines()[2:-1])

print(len(content))
with open(dst_file, 'w') as f:
    f.write('\n'.join(content))

df = pd.read_csv(dst_file, on_bad_lines='warn')
print(df.head())
print(df.info())
