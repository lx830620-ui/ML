# SMS Spam Detection — Machine Learning Project

A text classification project that detects spam SMS messages using two machine learning algorithms: **Support Vector Machine (SVM)** and a **from-scratch Naive Bayes classifier**.

---

## Project Overview

| Item | Detail |
|------|--------|
| **Task** | Binary text classification (Spam / Ham) |
| **Dataset** | UCI SMS Spam Collection (`spam.csv`, 5,574 messages) |
| **Algorithms** | SVM (Scikit-learn) + Naive Bayes (hand-implemented) |
| **Language** | Python 3 |
| **Key Libraries** | NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn |

---

## Project Structure

```
├── 源码/
│   ├── 支持向量机.py       # SVM classifier with TF-IDF features
│   └── 朴素贝叶斯.py       # Naive Bayes classifier (hand-implemented from scratch)
├── spam.csv                # Dataset
└── README.md
```

---

## Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- **Size**: 5,574 SMS messages
- **Class distribution**: ~87% Ham (legitimate), ~13% Spam (imbalanced)
- **Label encoding**: `ham → 0`, `spam → 1`

---

## Methodology

### Pipeline

```
Raw Text
   ↓
Data Loading & Preprocessing  (label encoding, column renaming)
   ↓
Feature Engineering            (TF-IDF vectorization, max 5000 features)
   ↓
Train / Test Split             (80% train, 20% test, random_state=42)
   ↓
Model Training                 (SVM / Naive Bayes)
   ↓
Evaluation                     (Accuracy, Precision, Recall, F1, Confusion Matrix)
```

### Algorithm 1 — SVM (`支持向量机.py`)

- Feature extraction via **TF-IDF** (top 5,000 terms, English stop words removed)
- **Linear kernel SVM** with `C=1.0`
- `class_weight='balanced'` to handle class imbalance
- Outputs: confusion matrix heatmap + per-class metrics

### Algorithm 2 — Naive Bayes (`朴素贝叶斯.py`)

- **Fully hand-implemented** without Scikit-learn's NB module
- Implements Bayes' theorem: $P(y|X) \propto P(y) \cdot \prod_i P(x_i|y)$
- **Laplace smoothing** ($\alpha = 1.0$) to handle unseen words
- Log-probability computation to avoid numerical underflow
- Custom `evaluate()` with classification report and confusion matrix

---

## Key Implementation Highlights

### Laplace Smoothing (Naive Bayes)
```python
# Prevents zero probability for unseen words
count = word_counts[cls].get(word, 0) + alpha
denominator = total_words_in_class + alpha * vocab_size
P(word | class) = count / denominator
```

### Log-Probability (Naive Bayes)
```python
# Avoids floating-point underflow from multiplying many small probabilities
log_prob = log(P(y)) + Σ log(P(word | y))
```

### TF-IDF (SVM)
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{DF}(t)}$$

- Assigns **higher weight** to words that are frequent in a document but rare overall
- Suppresses common words (e.g., "the", "is") automatically

---

## How to Run

```bash
# Clone or download the project
# Ensure spam.csv is in the same directory as the scripts

# Run SVM classifier
cd 源码
python 支持向量机.py

# Run Naive Bayes classifier
python 朴素贝叶斯.py
```

**Dependencies:**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## Example Predictions (SVM)

| Input Email | Prediction |
|-------------|-----------|
| "Free entry in 2 a wkly comp to win FA Cup final tkts!" | **SPAM** |
| "Hi John, are we still meeting tomorrow at 3pm?" | **HAM** |
| "Congratulations! Claim your prize now!" | **SPAM** |
| "Meeting reminder: Project review at 2pm in conference room B." | **HAM** |

---

## Results

> Run the scripts to generate exact metrics. Both models are evaluated with:
> - **Accuracy** — overall correctness
> - **Precision** — of predicted spam, how many are actually spam
> - **Recall** — of actual spam, how many are correctly identified
> - **F1 Score** — harmonic mean of Precision and Recall *(primary metric due to class imbalance)*
> - **Confusion Matrix** — visual breakdown of TP / TN / FP / FN

*Note: F1 Score is the primary metric here because the dataset is imbalanced (~87% Ham). Accuracy alone would be misleading.*

---

## What I Learned

- **TF-IDF** is a simple yet effective feature engineering technique for text classification
- **SVM with linear kernel** works well for high-dimensional sparse text features
- Implementing **Naive Bayes from scratch** deepened understanding of probabilistic modeling, Bayes' theorem, and numerical stability tricks (log-space computation)
- **Class imbalance** must be explicitly addressed — either via `class_weight='balanced'` or by choosing F1/Recall over Accuracy as the evaluation metric

---

