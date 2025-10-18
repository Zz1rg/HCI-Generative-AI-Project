#!/usr/bin/env python
import pandas as pd
from pathlib import Path

curr_dir = Path(__file__).parent.resolve()

with open(curr_dir / '../Data/combined_twitch_data.csv', 'r') as content_file:
    with open(curr_dir / '../Data/lowercase_twitch_data.txt', 'w') as f:
        f.write(content_file.read().lower())

df = pd.read_csv(curr_dir / '../Data/lowercase_twitch_data.txt', header=None)
df.drop_duplicates(inplace=True)
df.to_csv(curr_dir / '../Data/cleaned_twitch_data.txt', header=False, index=False)
