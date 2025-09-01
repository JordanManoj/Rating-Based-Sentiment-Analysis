# Sentiment Analysis Assignment (Rating-based version)
# -----------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

# --------------------------------------------------------
# Load Dataset
df = pd.read_csv("ratings_Electronics (1).csv")

# Rename columns for clarity
df = df.rename(columns={
    df.columns[0]: "reviewerID",
    df.columns[1]: "productID",
    df.columns[2]: "rating",
    df.columns[3]: "timestamp"
})

# --------------------------------------------------------
# Task 1: Preprocessing - Map Ratings to Sentiments
def rating_to_sentiment(rating):
    if rating <= 2.0:
        return "negative"
    elif rating == 3.0:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["rating"].apply(rating_to_sentiment)

# Show sample of preprocessed data
print("\n Preprocessed Data Sample (5 rows):\n")
print(df[["rating", "sentiment"]].head())

# --------------------------------------------------------
# Task 2: Feature Extraction
# Since no text is available, we'll use 'rating' as numeric feature
X = df[["rating"]].values  # Feature = rating
y = df["sentiment"]        # Target = sentiment labels

# --------------------------------------------------------
# Task 3: Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

# --------------------------------------------------------
# Task 4: Evaluation
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n {model_name} Results")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Recall:", recall_score(y_true, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    # Confusion Matrix
    labels = ["positive", "negative", "neutral"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate_model(y_test, lr_preds, "Logistic Regression")
evaluate_model(y_test, svm_preds, "SVM")

# --------------------------------------------------------
# --------------------------------------------------------
# Task 5: Visualization (Improved Word Clouds)

# Define sample words for each sentiment
positive_words = "good excellent amazing love satisfied happy recommend fantastic wonderful best"
negative_words = "bad terrible poor hate disappointed worst useless awful boring annoying"
neutral_words  = "okay average decent fine normal acceptable fair moderate mixed standard"

# Multiply these keywords by the number of reviews in each sentiment group
positive_text = " ".join([positive_words] * len(df[df['sentiment']=="positive"]))
negative_text = " ".join([negative_words] * len(df[df['sentiment']=="negative"]))
neutral_text  = " ".join([neutral_words]  * len(df[df['sentiment']=="neutral"]))

# Word Cloud - Positive
plt.figure(figsize=(10,5))
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap="Greens").generate(positive_text)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Positive Sentiment")
plt.show()

# Word Cloud - Negative
plt.figure(figsize=(10,5))
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap="Reds").generate(negative_text)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Negative Sentiment")
plt.show()

# Word Cloud - Neutral
plt.figure(figsize=(10,5))
wordcloud_neu = WordCloud(width=800, height=400, background_color='white', colormap="Blues").generate(neutral_text)
plt.imshow(wordcloud_neu, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Neutral Sentiment")
plt.show()