from datasets import load_dataset
import pandas as pd

# Download dataset
dataset = load_dataset("Arshia82sbn/youtube-sentiment-dataset")

# Convert training split to Pandas
df = dataset["train"].to_pandas()

# Save locally
df.to_csv("youtube_sentiment.csv", index=False, encoding="utf-8")

print("Dataset downloaded successfully!")
print("Shape:", df.shape)
print(df.head())
print(df["Sentiment"].value_counts())
