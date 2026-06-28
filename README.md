# Steam Reviews Sentiment Analysis
Social Media Computing 

## Overview
Full NLP pipeline on 500k Steam reviews (andrewmvd, Kaggle).
Covers VADER, TF-IDF + Logistic Regression/SVM/Random Forest, 
DistilBERT fine-tuning, and spaCy ABSA.

## Dataset
Steam Reviews by andrewmvd: https://www.kaggle.com/datasets/andrewmvd/steam-reviews
Download dataset.csv and place it in the same folder as Assignment.ipynb.
The dataset is not included here due to file size (6.4M rows).

## Environment
- Python 3.x, conda env: smc-gpu
- Key packages: pandas, numpy, nltk, scikit-learn, transformers, spacy, torch

## How to Run
1. Run Phase 1 cells (data loading through export of steam_phase1.csv)
2. Run Phase 2 cells (classical ML models)
3. Run Phase 3 cells on GPU (DistilBERT + ABSA)
