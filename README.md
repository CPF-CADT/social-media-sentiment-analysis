# Social Media Sentiment Analysis

A consent-based machine learning system that analyzes comments from authorized platform exports or APIs, classifying them as **positive, negative, or neutral** to help local businesses understand their audience.

## MVP

The current app is a local Streamlit workflow. It supports:

- CSV upload with a required `text` column and optional `label` column
- Human labeling of comments before training
- Word and character TF-IDF features for English, Khmer, and Romanized Khmer
- Logistic-regression sentiment classification
- Audience sentiment counts and single-text analysis
- Downloading the labeled dataset

It deliberately does not scrape Facebook pages, collect private data, or publish automated comments. A future connector should use an authorized Meta API integration for Pages the business owns or manages, with consent and App Review where required.

### Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

### Authorized Facebook browser collection

`facebook_reel_comments.py` contains a Selenium helper for an already-authenticated, authorized browser session. It selects **All comments**, expands visible replies, scrolls only the comments panel with a five-second pause, and writes exactly one CSV column named `comment`. Duplicate comment text is intentionally preserved.

The helper does not handle credentials, cookies, CAPTCHA solving, or posting. Example after you create your own Selenium driver and complete any login/CAPTCHA manually:

```python
from facebook_reel_comments import collect_reel_comments

rows = collect_reel_comments(driver, "comments.csv")
print(f"Saved {rows} comments")
```

If Chrome is already exposed through Selenium remote debugging and the Reel is open:

```bash
python run_comments.py --output comments.csv --pause 5
```

CSV example:

```csv
text,label
សេវាកម្មល្អណាស់,positive
ថ្លៃពេក,negative
ធម្មតា,neutral
```

## 1. Problem

Social media engagement metrics do not always reflect audience opinions. Manually analyzing large numbers of comments is time-consuming, especially with Khmer Unicode, Romanized Khmer, and mixed-language text.

## 2. Dataset

- **Sources:** Facebook, TikTok, Instagram, and YouTube
- **Data:** Comments collected through permitted APIs or other authorized methods
- **Target:** number of comments (to be decuss)
- **Labels:** Positive, Negative, Neutral

## 3. Machine Learning Approach

- **Task:** Supervised multiclass classification (3 classes)
- **Text representation:** TF-IDF or text embeddings
- **Algorithms:** To be determined
- **Evaluation:** Precision, Recall, F1-score

## 4. Expected Outcome

A web application that allows users to track supported social media posts, classify accessible comments, and visualize positive, negative, and neutral sentiment percentages.

## 5. Risks

- Platform API restrictions may limit comment collection.
- Romanized Khmer, slang, and sarcasm may reduce classification accuracy.
