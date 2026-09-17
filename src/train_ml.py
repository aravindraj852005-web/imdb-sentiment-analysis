import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import clean_text


print("Loading dataset...")

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))


print("\nCleaning training reviews...")
train_df["clean_review"] = train_df["review"].apply(clean_text)

print("Cleaning testing reviews...")
test_df["clean_review"] = test_df["review"].apply(clean_text)


print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train_df["clean_review"])
X_test = vectorizer.transform(test_df["clean_review"])

y_train = train_df["sentiment"]
y_test = test_df["sentiment"]

print("TF-IDF completed!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


print("\nTraining Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)

print("Naive Bayes Accuracy:", nb_accuracy)


print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_accuracy)


print("\nTraining SVM...")
svm = LinearSVC()
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_accuracy)


print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)


results = pd.DataFrame({
    "Model": [
        "Naive Bayes",
        "Logistic Regression",
        "SVM",
        "Random Forest"
    ],
    "Accuracy": [
        nb_accuracy,
        lr_accuracy,
        svm_accuracy,
        rf_accuracy
    ]
})

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results)


print("\nSaving models...")

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(nb, "models/naive_bayes.pkl")
joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(svm, "models/svm.pkl")
joblib.dump(rf, "models/random_forest.pkl")

results.to_csv("models/model_results.csv", index=False)

print("\nAll models saved successfully!")

print("\n==============================")
print("DETAILED MODEL EVALUATION")
print("==============================")

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_pred))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))