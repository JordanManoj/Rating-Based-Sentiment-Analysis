# Rating-Based Sentiment Analysis  

## Overview  
This project implements a sentiment analysis pipeline based on product **ratings** rather than review text. Using machine learning models, it classifies ratings into **positive, neutral, and negative sentiments**, evaluates performance, and visualizes results with charts and word clouds.  

---

## Problem Statement  
Sentiment analysis is a common application of NLP. However, when review text is missing, ratings can be mapped to sentiment classes. This project demonstrates how to process ratings, train classification models, and analyze results effectively.  

---

## Objectives  
1. Understand preprocessing steps on rating data.  
2. Map ratings to sentiment classes.  
3. Apply feature extraction.  
4. Train and evaluate ML models.  
5. Visualize insights with word clouds and charts.  

---

## Dataset  
- Source: Amazon Electronics Ratings Dataset.  
- Columns: `reviewerID`, `productID`, `rating`, `timestamp`.  
- Preprocessing:  
  - **Ratings mapped to sentiment**:  
    - 1–2 stars → Negative  
    - 3 stars → Neutral  
    - 4–5 stars → Positive  

---

## Tasks & Implementation  

### **Task 1: Data Preprocessing**  
- Loaded the dataset using Pandas.  
- Renamed columns for clarity.  
- Converted numerical ratings into sentiment labels.  
- Verified preprocessing with sample outputs (first 5 rows).  

### **Task 2: Feature Extraction**  
- Used ratings as numeric features.  
- Target variable: sentiment labels (positive, negative, neutral).  

### **Task 3: Model Building**  
- Split dataset into training (80%) and testing (20%).  
- Trained **Logistic Regression** and **SVM** models.  

### **Task 4: Evaluation**  
- Evaluated models using:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
- Visualized misclassifications with **confusion matrix**.  

### **Task 5: Visualization**  
- **Sentiment distribution** bar chart.  
- **Word clouds** generated using common keywords for each sentiment category (positive, negative, neutral).  

---

## How It Works  
1. Input dataset → ratings extracted.  
2. Ratings mapped to sentiment classes.  
3. Machine learning models trained to classify sentiment.  
4. Predictions evaluated using metrics.  
5. Visualizations created for insights.  

---

## Visualizations  
- Sentiment distribution bar chart.  
- Confusion matrices for both models.  
- Word clouds representing positive, negative, and neutral sentiments.  

---

## Results  
- Both models successfully classify ratings into sentiments.  
- Logistic Regression and SVM show high accuracy due to strong correlation between ratings and sentiment.  
- Visualizations provide quick insights into sentiment distribution.  

---


