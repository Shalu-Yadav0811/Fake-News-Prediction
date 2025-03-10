# Fake News Detection using Logistic Regression

## 📌 Overview
This project aims to detect **fake news articles** using **Logistic Regression**. The model is trained on a dataset containing real and fake news, utilizing **TF-IDF vectorization** for text preprocessing.

## 📂 Dataset
The dataset contains:
- **X (Features):** News articles converted into numerical format using **TF-IDF vectorization**.
- **Y (Labels):** Binary classification labels:
  - `1` → Fake news
  - `0` → Real news

## 🚀 Installation
To run this project, ensure you have **Python 3.11+** installed. Install the required dependencies using:

```bash
pip install numpy pandas scikit-learn
```

## 📜 Steps Involved
1. **Load Dataset**: Import and read the dataset.
2. **Text Preprocessing**:
   - Convert text to lowercase
   - Remove punctuation & stopwords
   - Apply stemming
   - Convert text to numerical form using **TF-IDF Vectorization**
3. **Split Data**: Training (80%) & Testing (20%)
4. **Train Model**: Using **Logistic Regression**
5. **Evaluate Model**:
   - Calculate **accuracy score**
   - Generate a **classification report**
6. **Make Predictions**: Detect whether a given news article is **Real or Fake**

## 🏗 Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Splitting dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Training Accuracy:', training_data_accuracy)
```

## 🔍 Prediction Example
```python
X_new = X_test[0]
prediction = model.predict(X_new)

if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')
```

## 📊 Results
- **Training Accuracy:** 98.98%
- **Testing Accuracy:** (Varies based on dataset)

## 📌 Future Improvements
- Implement **Deep Learning (LSTM, BERT)** for better accuracy.
- Add **real-world datasets** for generalization.
- Deploy the model as a **web application**.

##
🔥 **Built with Python & Scikit-Learn**

