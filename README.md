# Social Media Sentiment Analysis

A machine learning system that analyzes comments from **Facebook, TikTok, Instagram, and YouTube**, classifying them as **positive, negative, or neutral** to track audience sentiment.

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
