
import joblib
from pathlib import Path

MODEL_PATH = Path("models/spam_model.joblib")

if not MODEL_PATH.exists():
    print("Model not found. Run train_model.py first.")
    raise SystemExit(1)

# Load the trained model
model = joblib.load(MODEL_PATH)

print("Spam-Ham Classifier")
print("Type 'exit' to stop.")

while True:
    message = input("\nEnter an email or message: ").strip()

    if message.lower() == "exit":
        print("Program ended.")
        break

    if not message:
        print("Please enter a message.")
        continue

    prediction = model.predict([message])[0]

    print("Prediction:", prediction.upper())
