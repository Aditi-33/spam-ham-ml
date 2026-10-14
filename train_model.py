
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("email.csv", encoding="utf-8-sig")

# 2. Keep and rename the required columns
df = df[["Category", "Message"]].copy()
df.columns = ["label", "message"]

# 3. Clean the dataset
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].fillna("").astype(str).str.strip()

df = df[df["label"].isin(["ham", "spam"])]
df = df[df["message"] != ""]
df = df.drop_duplicates()

print("Dataset loaded successfully!")
print(df["label"].value_counts())

# 4. Separate features and target
X = df["message"]
y = df["label"]

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Create the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("classifier", MultinomialNB())
])

# 7. Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 8. Evaluate the model
y_pred = model.predict(X_test)

print("\nAccuracy:",
      round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 9. Save the trained model
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/spam_model.joblib")

print("\nModel saved successfully at models/spam_model.joblib")
