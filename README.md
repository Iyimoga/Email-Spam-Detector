# Email Spam Detector

> A classic NLP classifier that tells spam from legitimate email — built from scratch to understand how text classification actually works under the hood.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![NLP](https://img.shields.io/badge/NLP-scikit--learn-orange)

---

## What it does

Takes in raw email text and predicts whether it's **spam** or **ham** (legitimate). Trained on a labeled dataset, the model learns which words and patterns correlate with spam.

## The approach

Two-stage pipeline:

```
Raw email text
    ↓
Preprocessing (lowercase, strip punctuation, remove stopwords)
    ↓
TF-IDF vectorization (text → numerical features)
    ↓
Classifier (Naive Bayes / Logistic Regression)
    ↓
Spam / Ham prediction
```

## Why TF-IDF + Naive Bayes?

- **TF-IDF** (Term Frequency-Inverse Document Frequency) weighs words by how distinctive they are to a document — common words like "the" get low scores; spam-specific words like "winner" or "claim" get high ones.
- **Naive Bayes** is the textbook choice for text classification — fast, interpretable, and strong baseline. Great for understanding what's happening under the hood before reaching for deep learning.

## Tech stack

- **Python 3.10+**
- **scikit-learn** — vectorization + modeling
- **pandas** — data loading
- **NLTK** (optional) — stopwords and tokenization

## Running it

```bash
git clone https://github.com/Iyimoga/Email-Spam-Detector.git
cd Email-Spam-Detector
pip install -r requirements.txt
python train.py     # or open the notebook
```

## Example

```python
predict("Congratulations! You've won a $1000 gift card, click here...")
# → 'spam' (98% confidence)

predict("Hey, are we still meeting at 3pm tomorrow?")
# → 'ham' (99% confidence)
```

## Results

test set accuracy — 0.987

## What I learned

- Simple models with good preprocessing often beat complex ones
- Class imbalance matters — a dataset that's 90% ham makes "always predict ham" look deceptively accurate
- Feature engineering in NLP means caring about how text becomes numbers

## About

Built by **[Iyimoga Joseph Nana](https://github.com/Iyimoga)** — exploring NLP fundamentals one project at a time.
