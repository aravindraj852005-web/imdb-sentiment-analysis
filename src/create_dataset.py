import os
import pandas as pd

def load_reviews(folder, sentiment):
    reviews = []

    files = os.listdir(folder)
    total = len(files)

    for i, filename in enumerate(files, 1):
        filepath = os.path.join(folder, filename)

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                review = file.read()

            reviews.append({
                "review": review,
                "sentiment": sentiment
            })

        if i % 1000 == 0:
            print(f"Processed {i}/{total} reviews from {sentiment} folder")

    return reviews


print("Loading training data...")

train_pos = load_reviews("../dataset/aclImdb/train/pos", "positive")
train_neg = load_reviews("../dataset/aclImdb/train/neg", "negative")

print("Loading testing data...")

test_pos = load_reviews("../dataset/aclImdb/test/pos", "positive")
test_neg = load_reviews("../dataset/aclImdb/test/neg", "negative")


train_df = pd.DataFrame(train_pos + train_neg)
test_df = pd.DataFrame(test_pos + test_neg)


train_df.to_csv("../dataset/train.csv", index=False)
test_df.to_csv("../dataset/test.csv", index=False)


print("\nDataset created successfully!")
print("Training reviews:", len(train_df))
print("Testing reviews:", len(test_df))

print("\nTraining sentiment distribution:")
print(train_df["sentiment"].value_counts())

print("\nTesting sentiment distribution:")
print(test_df["sentiment"].value_counts())