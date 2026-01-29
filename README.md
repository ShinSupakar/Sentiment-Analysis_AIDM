# Sentiment Analysis: Three-Method Comparison (TF-IDF, Frozen BERT, Fine-tuned BERT)

## 📌 Overview

This project addresses binary sentiment classification of product reviews (positive vs. negative) as part of IE4483 – Artificial Intelligence & Data Mining (Mini Project).

The core objective is to compare three sentiment analysis methodologies, progressing from classical machine learning to deep learning with transformers, and to analyze their performance, computational cost, and robustness.

The three methods implemented and evaluated are:

TF-IDF + SMOTE + Logistic Regression (baseline)

Frozen BERT embeddings + Logistic Regression (transfer learning)

Fine-tuned BERT neural network (end-to-end deep learning)

## 📊 Methods Implemented

This project implements and evaluates three distinct sentiment analysis approaches. These are all present in the code (see MiniProject_Code.ipynb) and are used for comparative analysis.

### Method 1: TF-IDF + Logistic Regression (Baseline)

Feature Extraction: TF-IDF vectorization (max 5000 features)

Class Imbalance Handling: SMOTE oversampling

Purpose: Provides a lightweight baseline for comparison

Pros: Fast, low resource usage

Cons: No contextual understanding of language

### Method 2: Frozen BERT Feature Extraction

Backbone: bert-base-uncased

Approach:

- BERT is used only as a feature extractor

- All BERT weights are frozen

- Extracted embeddings are fed into a downstream classifier

Pros: Captures semantic information with reduced training cost

Cons: Less flexible than full fine-tuning

### Method 3: Fine-Tuned BERT Neural Network

Backbone: bert-base-uncased

Approach:

- End-to-end fine-tuning of BERT on the sentiment task

- Tokenization with attention masks

- Neural classifier trained jointly with BERT

Pros: Best performance due to task-specific adaptation

Cons: Computationally expensive


## ⚙️ Data Preprocessing

Two preprocessing strategies are used depending on the model:

- **Classical ML (TF-IDF):**
  - Lowercasing
  - Removal of punctuation and digits
  - Stopword handling via TF-IDF

- **BERT-based models:**
  - Minimal preprocessing to preserve context
  - Tokenization using BERT WordPiece tokenizer
  - Padding and truncation to fixed length (128 tokens)

## 📉 Training and Hyperparameters

| Model            | Key Settings                                     |
|------------------|--------------------------------------------------|
| TF-IDF + LR      | max_features=5000, SMOTE, max_iter=1000          |
| Frozen BERT      | CLS embeddings (768-dim), frozen encoder         |
| Fine-tuned BERT  | AdamW, lr=2e-5, batch=16, epochs=4               |

Hyperparameters were chosen based on literature best practices and empirical validation.

## 🏁 Evaluation and Results

- Metrics used: Accuracy, F1-score

- **Best performance**: Fine-tuned BERT
  - **Accuracy** ≈ 95.48%
  - **F1-score** ≈ 97.36%

## 🧪 Extensions and Future Work

- Domain adaptation (e.g., hotel or movie reviews)

- Multi-class sentiment classification (1–5 stars)

- Robust learning under noisy labels

- Improved handling of mixed and contrastive sentiment
