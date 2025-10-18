#!/usr/bin/env python
# vim: sw=2
from google import genai
from google.genai import types
from pathlib import Path
from itertools import islice
import time
import os

curr_dir = Path(__file__).parent.resolve()

def generate():
  client = genai.Client(api_key=os.environ.get("API_KEY"))

  with open(curr_dir / "../Data/converted_hate_speech_detection_curated_dataset/sampled_100.csv", "rb") as f:
    labeled = f.read()
  with open(curr_dir / "../Data/cleaned_twitch_data.txt", "r") as f:
    unlabeled = f.read().splitlines()

  msg_text1 = types.Part.from_text(text="""You will be provided with a labeled dataset in CSV format and a file containing unlabeled messages.

Labeled Dataset (CSV format with 'message' and 'label' columns, where 0 = not offensive, 1 = offensive):""")
  msg_document1 = types.Part.from_bytes(
    data=labeled,
    mime_type="text/csv",
  )
  msg_text2 = types.Part.from_text(text="""Follow these instructions precisely to complete the task:
1.  **Analyze the Labeled Dataset:** First, carefully study the labeled dataset. Identify the patterns, keywords, phrases, and context that distinguish offensive messages (label 1) from non-offensive messages (label 0).
2.  **Classify Unlabeled Messages:** For each message in the unlabeled messages, classify it as either offensive or not offensive based *only* on the patterns you learned from the labeled dataset.
3.  **Assign Labels:**
    *   Assign a label of `1` if the message is offensive.
    *   Assign a label of `0` if the message is not offensive.
4.  **Format the Output:** Present the final result in a CSV format with a header row. The CSV must contain two columns: `message` and `label`.

**Example:**

If you are provided with the following inputs:

**Labeled Dataset:**
```csv
message,label
you are a total idiot, uninstall the game,1
ggwp everyone, that was a fun match,0
I hope you get banned for that move,1
nice shot!,0
```

**Unlabeled Messages:**
```
what a loser
good game
```

**Your output must be:**
```csv
message,label
what a loser,1
good game,0
```""")

  si_text1 = """You are an expert Data Scientist specializing in Natural Language Processing (NLP) and content moderation. Your task is to analyze a labeled dataset of messages to learn what constitutes offensive content, and then use that knowledge to classify a new set of unlabeled messages."""

  it = iter(unlabeled)
  n = 50
  for x in [list(islice(it, n)) for _ in range((len(unlabeled) + n - 1) // n)]:
    print(f'---\n{x}\n---')
    model = "gemini-2.5-flash-preview-09-2025"
    contents = [
      types.Content(
        role="user",
        parts=[
          msg_text1,
          msg_document1,
          msg_text2,
          types.Part.from_text(text=f"""Unlabeled Messages (one message per line):
```
{'\n'.join(x)}
```"""),
        ]
      ),
    ]

    generate_content_config = types.GenerateContentConfig(
      temperature = 1,
      top_p = 0.95,
      max_output_tokens = 300000,
      safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
      ), types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
      ), types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
      ), types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
      )],
      system_instruction=[types.Part.from_text(text=si_text1)],
      thinking_config=types.ThinkingConfig(
        thinking_budget=-1,
      ),
    )

    with open(f'output-{time.time()}.txt', 'w') as f:
      for chunk in client.models.generate_content_stream(
        model = model,
        contents = contents,
        config = generate_content_config,
        ):
        if chunk.text is not None:
          print(chunk.text, end="")
          print(chunk.text, end="", file=f)

generate()
