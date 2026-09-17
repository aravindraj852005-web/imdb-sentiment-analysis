import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Create output folder
os.makedirs("eda_outputs", exist_ok=True)

# Load dataset
train_df = pd.read_csv("dataset/train.csv")

print("Dataset loaded successfully!")
print("Total reviews:", len(train_df))

# ==========================================
# 1. SENTIMENT DISTRIBUTION
# ==========================================

sentiment_counts = train_df["sentiment"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values)

plt.title("IMDb Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()
plt.savefig(
    "eda_outputs/sentiment_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# 2. REVIEW LENGTH DISTRIBUTION
# ==========================================

train_df["review_length"] = train_df["review"].str.split().str.len()

plt.figure(figsize=(8, 5))

for sentiment in ["positive", "negative"]:
    data = train_df[
        train_df["sentiment"] == sentiment
    ]["review_length"]

    plt.hist(
        data,
        bins=50,
        alpha=0.5,
        label=sentiment
    )

plt.title("Review Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Number of Reviews")
plt.legend()

plt.tight_layout()
plt.savefig(
    "eda_outputs/review_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# 3. POSITIVE WORDCLOUD
# ==========================================

positive_text = " ".join(
    train_df[
        train_df["sentiment"] == "positive"
    ]["review"].astype(str)
)

positive_wc = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(12, 6))
plt.imshow(positive_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews WordCloud")

plt.tight_layout()
plt.savefig(
    "eda_outputs/positive_wordcloud.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# 4. NEGATIVE WORDCLOUD
# ==========================================

negative_text = " ".join(
    train_df[
        train_df["sentiment"] == "negative"
    ]["review"].astype(str)
)

negative_wc = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(negative_text)

plt.figure(figsize=(12, 6))
plt.imshow(negative_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews WordCloud")

plt.tight_layout()
plt.savefig(
    "eda_outputs/negative_wordcloud.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


print("\nEDA completed successfully!")
print("Charts saved inside: eda_outputs/")