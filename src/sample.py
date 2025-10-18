#!/usr/bin/env python
import pandas as pd
from pathlib import Path

curr_dir = Path(__file__).parent.resolve()
random_state = 1
df = pd.read_csv(curr_dir / '../Data/converted_hate_speech_detection_curated_dataset/data_balanced.csv')
df = df[['message', 'label']]
print(df.info())

n = 50
sampled = pd.concat([
    df[df['label'] == 0].sample(n=n, random_state=random_state),
    df[df['label'] != 0].sample(n=n, random_state=random_state),
])
sampled.reset_index(drop=True, inplace=True)
print(sampled.head(20))
print(sampled.info())
print(sampled['label'].value_counts())
sampled.to_csv(curr_dir / '../Data/converted_hate_speech_detection_curated_dataset/sampled_100.csv', index=False)
