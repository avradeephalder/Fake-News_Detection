# 📰 AI Fake News Detection System

End-to-end fake news detection application using Natural Language Processing and Ensemble Machine Learning. Real-time article classification into Real/Fake with confidence scores. Built with Python, NLTK, scikit-learn, and TF-IDF.

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange)

---

## ✨ Key Features

- 📰 **Real-time Detection:** Instant fake news classification from article text
- 🤖 **Ensemble Learning:** VotingClassifier combining Logistic Regression, Random Forest, and Decision Tree
- 🎯 **High Accuracy:** TF-IDF vectorization with ensemble methods (~96% accuracy)
- 📊 **Robust Predictions:** Hard voting for stable, consistent results
- ⚡ **Fast Inference:** Sub-second prediction on CPU
- 🧹 **Text Preprocessing:** NLTK-based cleaning with stopword removal and lemmatization
- 💾 **Modular Pipeline:** Clean separation of preprocessing, vectorization, and modeling layers
- 📈 **Performance Metrics:** Accuracy, F1-score, precision, recall, and confusion matrix visualization

---

## 📚 Tech Stack

### Machine Learning
- **Python 3.8+**
- **scikit-learn** for ML models and vectorization
- **NLTK** for text preprocessing (stopwords, WordNetLemmatizer)
- **pandas** & **numpy** for data manipulation
- **seaborn** & **matplotlib** for visualization
- **joblib** for model serialization

### Models
- **TF-IDF Vectorizer** (uni-grams + bi-grams, max 5000 features)
- **Logistic Regression** (linear baseline classifier)
- **Random Forest** (200 estimators, bagging ensemble)
- **Decision Tree** (non-linear decision boundaries)
- **VotingClassifier** (hard voting ensemble)

---

## 🏆 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | VotingClassifier (LR + RF + DT) |
| **Training Dataset** | ISOT Fake News Dataset (~44,000 articles) |
| **Features** | TF-IDF (uni-grams + bi-grams) |
| **Classes** | Binary (Real / Fake) |
| **Validation Accuracy** | ~96% |
| **F1-Score** | ~0.96 |
| **Inference Time** | < 200ms per article |

---

## 📦 Installation

### Prerequisites
- **Python** (v3.8-3.12)
- **Google Colab** (recommended for training)
- Datasets: `True.csv` and `Fake.csv` (ISOT Fake News Dataset)

### 1. Clone the Repository

```
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies

```
pip install pandas numpy nltk scikit-learn seaborn matplotlib joblib
```

### 3. Download NLTK Data

```
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 4. Prepare Datasets

Place your datasets in the project directory:
- `True.csv` - Real news articles
- `Fake.csv` - Fake news articles

---

## 🖥️ Usage

### Training the Model (Google Colab)

Open the notebook `fake_news_detection.ipynb` in Google Colab and run all cells:

1. **Mount Google Drive** and load datasets
2. **Preprocess text** with NLTK (stopwords + lemmatization)
3. **Vectorize** with TF-IDF (max 5000 features, uni/bi-grams)
4. **Train ensemble** with VotingClassifier
5. **Evaluate** with accuracy, F1, and confusion matrix
6. **Save models** (ensemble + vectorizer) to Drive

### Making Predictions

```
def predict_news(article):
    """Classify article as Real or Fake"""
    cleaned = clean_text(article)
    vectorized = tfidf.transform([cleaned])
    prediction = ensemble.predict(vectorized)
    return "Fake News ❌" if prediction == 1 else "Real News ✅"

# Example
article = "Breaking: Scientists discover new renewable energy source..."
print(predict_news(article))  # Output: Real News ✅
```

---

## 🏗️ Project Structure

```
fake-news-detection/
│
├── notebooks/
│   └── fake_news_detection.ipynb    # Training notebook (Colab)
│
├── data/
│   ├── True.csv                      # Real news dataset
│   └── Fake.csv                      # Fake news dataset
│
├── models/                           # Saved models (after training)
│   ├── ensemble_model.pkl            # Trained VotingClassifier
│   └── tfidf_vectorizer.pkl          # Fitted TF-IDF vectorizer
│
├── requirements.txt                  # Python dependencies
├── LICENSE                           # Apache 2.0 License
└── README.md                         # This file
```

---

## 🔧 Pipeline Details

### 1. Text Preprocessing

```
def clean_text(text):
    """NLTK-based text cleaning"""
    text = text.lower()  # Lowercase
    text = ''.join([c for c in text if c not in string.punctuation])  # Remove punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords]  # Lemmatize & remove stopwords
    return ' '.join(tokens)
```

### 2. TF-IDF Vectorization

```
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['clean_text'])
```

### 3. Ensemble Training

```
rf = RandomForestClassifier(n_estimators=200, random_state=42)
dt = DecisionTreeClassifier(random_state=42)
lr = LogisticRegression(max_iter=300, random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('dt', dt), ('lr', lr)],
    voting='hard'
)
ensemble.fit(X_train, y_train)
```

### 4. Evaluation

```
y_pred = ensemble.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Real','Fake']))
```

---

## 🌟 Key Features Explained

### NLP Pipeline
- **Stopword Removal:** Eliminates common words (the, is, at, etc.) for cleaner features
- **Lemmatization:** Converts words to root form (running → run) for consistency
- **TF-IDF:** Weighs word importance by frequency and rarity across documents
- **N-grams:** Captures uni-grams (single words) and bi-grams (word pairs)

### Ensemble Learning
- **Logistic Regression:** Fast linear baseline with good interpretability
- **Random Forest:** Handles non-linear patterns with bagging for variance reduction
- **Decision Tree:** Captures complex decision boundaries
- **Hard Voting:** Majority vote from all three models for robust predictions

### Performance Visualization
- **Confusion Matrix:** Visual heatmap showing true/false positives/negatives
- **Classification Report:** Per-class precision, recall, F1-score
- **Accuracy Score:** Overall correctness percentage

---

## 🚀 Training Your Own Model

### Dataset Preparation

```
import pandas as pd

# Load datasets
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# Add labels
true_df['label'] = 0  # Real
fake_df['label'] = 1  # Fake

# Combine and shuffle
df = pd.concat([true_df, fake_df], axis=0).sample(frac=1).reset_index(drop=True)
```

### Model Training

```
from sklearn.model_selection import train_test_split

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ensemble
ensemble.fit(X_train, y_train)

# Save
import joblib
joblib.dump(ensemble, 'ensemble_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
```

---

## ⚡ Performance Tips

### Speed Optimization
1. **Reduce max_features:** Use 2000-3000 instead of 5000 for faster vectorization
2. **Use only uni-grams:** Set `ngram_range=(1,1)` to skip bi-grams
3. **Simplify ensemble:** Remove Decision Tree for faster predictions

### Accuracy Improvement
1. **More training data:** Add datasets like LIAR, FakeNewsNet, FEVER
2. **Hyperparameter tuning:** Optimize n_estimators, max_depth, C values
3. **Advanced models:** Try BERT, RoBERTa for 98%+ accuracy
4. **Feature engineering:** Add metadata (source, author, date) as features

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Avradeep Halder**

- LinkedIn: [linkedin.com/in/avradeephalder](https://www.linkedin.com/in/avradeephalder/)
- GitHub: [@avradeephalder](https://github.com/avradeephalder)

---

## 🙏 Acknowledgments

- [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/) for training data
- [scikit-learn](https://scikit-learn.org/) for machine learning framework
- [NLTK](https://www.nltk.org/) for natural language processing
- [Google Colab](https://colab.research.google.com/) for free GPU/TPU resources

---

## 🐛 Troubleshooting

### Model doesn't load
- Ensure `ensemble_model.pkl` and `tfidf_vectorizer.pkl` are in the correct directory
- Check scikit-learn version compatibility (`pip install --upgrade scikit-learn`)

### NLTK data not found
- Run `nltk.download('stopwords')` and `nltk.download('wordnet')`
- Ensure NLTK data path is correct

### Low accuracy
- Check if datasets are properly labeled (0=Real, 1=Fake)
- Ensure text preprocessing is applied consistently
- Try increasing max_features or adding more training data

### Prediction errors
- Verify input text is a string
- Check if `clean_text()` function is defined
- Ensure models are loaded before calling `predict_news()`

---

## 📧 Contact

For questions or support, please [open an issue](https://github.com/yourusername/fake-news-detection/issues) or contact me via GitHub.

---

**⭐ If you find this project helpful, please give it a star!**
```

***

### To Add This README to GitHub:

**Option 1: Via GitHub Web Interface**
1. Go to your repository on GitHub
2. Click **Add file** → **Create new file**
3. Name it `README.md`
4. Paste the content above
5. Scroll down, add commit message: "docs: add comprehensive README"
6. Click **Commit changes**

**Option 2: Via Colab (if saving notebook)**
1. In Colab: **File** → **Save a copy in GitHub**
2. Select your repo
3. Also create `README.md` separately via GitHub web

This README matches your sentiment analysis style with badges, detailed sections, and professional formatting! 📰✨
