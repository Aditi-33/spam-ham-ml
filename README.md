# Spam-Ham Message Classifier using Machine Learning

## About the Project

This project is a Machine Learning-based Spam-Ham Message Classifier that identifies whether a text message is spam or legitimate (ham).

The model learns patterns from labelled messages and predicts the category of a new message entered by the user.

## Features

- Classifies messages as Spam or Ham.
- Uses TF-IDF for text feature extraction.
- Uses Multinomial Naive Bayes for classification.
- Evaluates model performance using accuracy and a classification report.
- Saves the trained model for future predictions.
- Allows users to test custom messages through the terminal.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Joblib

## Project Structure

```text
spam-ham-ml/
├── train_model.py
├── spam_classifier.py
├── email.csv
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── spam_model.joblib
```

The `models/` folder and its saved model are generated after training. The saved model does not need to be committed to GitHub because it can be recreated.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd spam-ham-ml
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Make sure `email.csv` is present in the project directory.

```bash
python train_model.py
```

This trains the classifier, displays evaluation metrics, and saves the model to `models/spam_model.joblib`.

### 4. Run the classifier

```bash
python spam_classifier.py
```

Enter a message when prompted. The program predicts whether it is Spam or Ham.

Type `exit` to stop the program.

## How It Works

1. **Data Loading:** Loads labelled messages from the CSV dataset.
2. **Data Preprocessing:** Cleans the labels and removes empty or duplicate records.
3. **Train-Test Split:** Divides the dataset into training and testing sets.
4. **Feature Extraction:** TF-IDF converts text into numerical features.
5. **Model Training:** Multinomial Naive Bayes learns patterns from the training messages.
6. **Evaluation:** The model is evaluated using accuracy, precision, recall, and F1-score.
7. **Prediction:** The trained model classifies new messages as Spam or Ham.

## Dataset

## Dataset

The dataset used in this project was obtained from Kaggle. It contains text messages labelled as `spam` or `ham`, which are used to train and evaluate the machine learning model.

**Dataset source:** Kaggle

[View Kaggle Datasets](https://www.kaggle.com/datasets)


## Example

Enter an email or message: Congratulations! You won a free prize!

Prediction: SPAM

Predictions may vary depending on the training data and message content.

## Future Improvements

- Develop a web interface for message classification.
- Compare Naive Bayes with other classification algorithms.
- Add visualizations of model performance.
- Improve classification using additional training data.

