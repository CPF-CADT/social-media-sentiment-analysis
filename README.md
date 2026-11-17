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

## Project Roadmap

The project follows these steps to develop a YouTube comment sentiment analysis system:

1. **Data Collection**
   - Method: YouTube Data API.
   - Collect video metadata, comments, video IDs, and timestamps.

2. **Data Labeling**
   - Method: LLM-assisted labeling with human verification.
   - Assign three sentiment classes: Positive, Negative, and Neutral.
   - Label original comments before text cleaning.

3. **Data Preprocessing**
   - Method: NLP text cleaning.
   - Apply lowercasing, URL and noise removal, tokenization, and stopword removal while preserving meaningful sentiment expressions.

4. **Feature Extraction**
   - Method: TF-IDF Vectorization.
   - Convert cleaned comments into numerical feature vectors.

5. **Model Training**
   - Method: Supervised Machine Learning.
   - Train and compare Naïve Bayes, Logistic Regression, and SVM using an 80/20 train-test split.

6. **Model Evaluation**
   - Method: Classification Metrics.
   - Evaluate using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

7. **Dashboard Development**
   - Method: FastAPI, HTML, CSS, and data visualization.
   - Display sentiment distribution, individual video analysis, and overall audience sentiment.
