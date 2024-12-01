# Twitter Sentiment Analysis

## Overview
This project implements a machine learning approach to analyze sentiments in tweets using the Sentiment140 dataset from Kaggle. The analysis includes various text preprocessing techniques, feature extraction, dimensionality reduction, and multiple classification models to achieve accurate sentiment prediction.

## Features
- Text preprocessing and cleaning
- TF-IDF vectorization for feature extraction
- Dimensionality reduction techniques (PCA, t-SNE, UMAP)
- Multiple classification models (Naive Bayes, Logistic Regression)
- Advanced visualization of results
- Feature importance analysis
- Model performance comparison

## Dependencies
```
numpy
pandas
scikit-learn
matplotlib
seaborn
umap-learn
wordcloud
scipy
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset
The project uses the Sentiment140 dataset from Kaggle (https://www.kaggle.com/datasets/kazanova/sentiment140) which contains:
- 1.6 million tweets
- Binary sentiment classification (0 for negative, 4 for positive)
- Features: target, id, date, flag, user, and text

## Project Structure
The main components include:
- Data preprocessing and cleaning
- Feature extraction using TF-IDF
- Dimensionality reduction analysis
- Model training and evaluation
- Visualization of results

## Model Performance
The project implements and compares multiple models:
- Naive Bayes (Baseline model)
- Logistic Regression
- Logistic Regression with Feature Selection

## Visualizations
The project includes various visualization techniques:
- Word clouds for positive and negative sentiments
- Feature importance plots
- ROC curves
- Correlation heatmaps
- Clustering visualizations

## Usage
1. Load and preprocess the data:
```python
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding='ISO-8859-1',
                 names=['target', 'id', 'date', 'flag', 'user', 'text'])
```

2. Run text preprocessing:
```python
df['processed_text'] = df['text'].apply(preprocess_text)
```

3. Train and evaluate models:
```python
evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
```

## Results
- Feature selection significantly improved model performance
- Logistic Regression outperformed Naive Bayes
- Dimensionality reduction maintained performance while improving efficiency

## Contributing
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request


## Authors
- Pushkar Visave
- Saurabh Dighe
- Vibhav Rane

## Acknowledgments
- Sentiment140 dataset creators
- Kaggle community
- Course instructor: Oleksandr Narykov
