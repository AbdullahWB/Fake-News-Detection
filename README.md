<!-- Replace this with your own banner image -->
<p align="center">
  <img src="assets/Fake News Detection.png" alt="Lightweight Fake News Detection" width="100%">
</p>

<h1 align="center">Lightweight Fake News Detection on ISOT & WELFake</h1>

<p align="center">
  <b>Bachelor Thesis / Research Project</b><br>
  Linguistic Features • Classical ML • Cross-Dataset Evaluation
</p>

---

## 🌍 Project Overview

This repository contains my bachelor research project on **fake news detection** using two widely used benchmark datasets:

- **ISOT / Fake & Real News dataset** – True.csv + Fake.csv with title, full text, subject, and date for each article.
- **WELFake dataset** – 72,134 articles (35,028 real, 37,106 fake) built by merging multiple popular fake-news corpora.

The main goals of this project are:

- ✅ Build **strong classical baselines** using TF-IDF features and linear models
- ✅ Implement a **WELFake-inspired** pipeline combining:
  - **Linguistic / stylistic features (LFS)**
  - **Bag-of-words** representations (TF-IDF, CountVectorizer)
- ✅ Evaluate **tri-view ensembles** (TF-IDF + CV + LFS+CV) on both ISOT & WELFake
- ✅ Study **cross-dataset generalization**: train on one dataset, test on the other

This is a **research prototype** focused on methodology, interpretability, and low computational cost.  
It is **not** a production misinformation detection system or a replacement for human fact-checking.

---

## 📦 Datasets Used

> ⚠️ Raw datasets are **not** included in this repository.  
> To reproduce experiments, please download them from their official sources and update the paths in the notebooks.

1. **ISOT / Fake & Real News Dataset**

   - Files: `True.csv`, `Fake.csv`
   - Each article has:
     - Title
     - Full body text
     - Subject (e.g., politics, world news)
     - Publication date
   - Use in this project:
     - Text-only fake vs real classification
     - In-dataset evaluation and cross-dataset experiments

2. **WELFake Dataset** (Word Embedding over Linguistic Features for Fake News Detection)
   - 72,134 news articles:
     - 35,028 real
     - 37,106 fake
   - Built by merging four popular sources (Kaggle, McIntire, Reuters, BuzzFeed Political) to provide a larger and more diverse corpus for fake news detection.
   - Columns:
     - `title`, `text`, `label` (0 = fake, 1 = real)
   - Use in this project:
     - Large-scale training for classical models
     - Evaluation of WELFake-style feature fusion
     - Cross-dataset experiments with ISOT

---

## 🧪 Methodology

The project is structured like a small **experimental study** carefully following and extending the WELFake methodology.

### 1️⃣ Preprocessing & Exploratory Analysis

For both datasets:

- Inspect class balance, title lengths, and article lengths
- Perform light text normalization:
  - Lowercasing
  - Removal of URLs and HTML tags
  - Removal of non-letter characters
  - Whitespace normalization
- Create cleaned fields: `title_clean`, `text_clean`

This step is implemented in `notebooks/01_eda.ipynb`.

---

### 2️⃣ Feature Engineering

The project uses **three main feature families**:

#### 📝 2.1 Bag-of-Words Text Features

- **TF-IDF** (unigrams + bigrams) on `text_clean`
  - Limit `max_features` to keep compute manageable
  - Remove English stopwords
- **CountVectorizer (CV)** (unigrams + bigrams)
  - Mirrors the bag-of-words setup from many WELFake-based works

These serve as strong classical baselines for text-only fake news detection.

#### 🔍 2.2 Linguistic / Stylistic Features (LFS)

Inspired by the WELFake paper’s linguistic feature sets, the project extracts a compact but expressive set of **hand-crafted features** from the raw article text:

- **Quantity & style**
  - Total word count
  - Number of sentences
  - Average sentence length (words per sentence)
- **Surface cues**
  - Number of exclamation marks (`!`)
  - Number of question marks (`?`)
  - Ratio of uppercase letters
- **Readability**
  - Flesch Reading Ease
  - SMOG Index
  - Gunning Fog Index (via `textstat`)
- **Sentiment**
  - Polarity and subjectivity (via `TextBlob`)

All LFS are standardized (z-score) before modeling.

#### 🧬 2.3 Fusion Features (LFS + CV)

To emulate the **“linguistic + word features”** idea from WELFake:

- Concatenate LFS vectors with CountVectorizer representations:
  \[
  X\_{\text{LFS+CV}} = [\text{LFS} \,\|\, \text{CV}]
  \]
- Use this fused space as a richer representation of each news article.

---

### 3️⃣ Modeling & Evaluation

All models are implemented in scikit-learn for **simplicity and reproducibility**.

#### ⚙️ 3.1 Base Models

For each dataset and each feature configuration (`TF-IDF`, `CV`, `LFS`, `LFS+CV`):

- **Logistic Regression**
  - `solver="liblinear"`, `class_weight="balanced"`
  - Hyperparameter: `C`, `penalty ∈ {l1, l2}`
- **Linear SVM (LinearSVC)**
  - `class_weight="balanced"`
  - Hyperparameter: `C`
- **Random Forest**
  - `class_weight="balanced_subsample"`
  - Hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`

Hyperparameters are tuned via **RandomizedSearchCV** with stratified 3-fold cross-validation.

#### 📊 3.2 Evaluation Protocol

- 80/20 **stratified train/test split** for:
  - ISOT only
  - WELFake only
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Confusion matrices for qualitative error analysis

All results are saved to CSV in the `results/` directory.

---

### 4️⃣ WELFake-Style Tri-View Ensemble

Inspired by the WELFake framework’s multi-view voting scheme, the project builds a **hard-voting ensemble** over three complementary feature spaces:

1. Best model on **TF-IDF (text)**
2. Best model on **CV (text)**
3. Best model on **LFS+CV (fusion)**

The ensemble outputs the **majority vote** across these three models, providing a robust, low-cost alternative to heavier deep-learning architectures.

---

### 5️⃣ Cross-Dataset Generalization

To test **robustness and generalization**, the pipeline includes:

1. **Train on ISOT → Test on WELFake**
2. **Train on WELFake → Test on ISOT**

For each direction:

- Select the _single best_ model + feature space (by F1-score on the source dataset)
- Train on the **entire source** dataset
- Evaluate on the **entire target** dataset

This cross-dataset setup goes beyond most single-dataset fake-news studies and is intended to be a key part of the research contribution.

---

## ⚙️ Lightweight & Reproducible Design

The entire pipeline is designed to be:

- **Lightweight**: uses TF-IDF, bag-of-words, and hand-crafted features with classical ML models instead of large transformers or deep networks.
- **Reproducible**:
  - Fixed random seeds
  - Clear train/test splits
  - Explicit feature extraction steps
  - Metrics stored as CSV in `results/`
- **CPU-friendly**: all experiments are runnable on a standard laptop / Colab CPU instance.

---

## 📂 Repository Structure (example)

> ⚠️ Adjust this section to match your actual file/folder names.

```text
.
├── data/
│   ├── True.csv                 # ISOT real news (not tracked)
│   ├── Fake.csv                 # ISOT fake news (not tracked)
│   └── WELFake.csv              # WELFake dataset (not tracked)
├── notebooks/
│   ├── 01_eda.ipynb             # Data loading, cleaning, EDA
│   ├── 02_features_baselines.ipynb
│   │                             # TF-IDF, CV, metadata/LFS, baseline models
│   └── 03_welfake_style_pipeline.ipynb
│                                 # LFS extraction, fusion, ensembles, cross-dataset
├── src/
│   ├── preprocessing.py          # Text cleaning & utilities
│   ├── features_lfs.py           # Linguistic / stylistic feature extraction
│   ├── vectorizers.py            # TF-IDF and CountVectorizer builders
│   ├── models.py                 # Model definitions & hyperparameter search
│   └── evaluation.py             # Metrics, confusion matrices, result saving
├── results/
│   ├── baseline_and_ensemble_metrics.csv
│   └── cross_dataset_metrics.csv
├── models/                       # Saved .joblib models
├── figures/                      # Plots, confusion matrices
├── assets/
│   └── banner.png                # README banner image
├── README.md
└── requirements.txt
```
