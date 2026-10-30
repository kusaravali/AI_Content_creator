# AI_Content_creator

Generates social media content (title, hook, caption, video script,
hashtags, call to action) using the open-source
`HuggingFaceTB/SmolLM2-360M-Instruct` model. No paid API keys needed.

## Features
- Content package for any topic, platform, tone and language
- Optional 5 extra post ideas
- Saves output to `student_ai_content.txt`
- Automatic GPU/CPU detection

## Installation
```
pip install -r requirements.txt
```

## Usage
```
python content_creator.py
```

## Tech Stack
Python, PyTorch, Hugging Face Transformers
