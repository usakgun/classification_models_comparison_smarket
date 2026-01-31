# Stock Market Direction Prediction

This project applies various machine learning classification techniques to predict the direction of the S&P 500 stock index (Up/Down) using the "Smarket" dataset. It compares the performance of Logistic Regression, LDA, QDA, and KNN algorithms.

## Project Overview
The goal is to predict whether the market will go up or down on a given day based on the percentage returns of the five previous days (Lag1-Lag5) and the trading volume. The project explores feature selection and compares linear vs. non-linear decision boundaries to find the most effective model.

## Methodology
The analysis proceeds in several steps:
1.  **Exploratory Data Analysis (EDA):** Correlation analysis to understand relationships between lag variables and volume.
2.  **Data Splitting:**
    * **Training Set:** Data from 2001 to 2004.
    * **Test Set:** Data from 2005.
3.  **Model Implementation:**
    * **Logistic Regression:** Baseline model (tested with all features vs. selected features).
    * **Linear Discriminant Analysis (LDA):** Assumes Gaussian distribution with shared covariance.
    * **Quadratic Discriminant Analysis (QDA):** Assumes Gaussian distribution with class-specific covariance.
    * **K-Nearest Neighbors (KNN):** Non-parametric approach (tested with K=1 and K=3).

## Performance Results
The models were evaluated based on prediction accuracy on the test data (Year 2005).

* **QDA (Quadratic Discriminant Analysis):** ~60% Accuracy. This was the best performing model, suggesting a non-linear relationship in the data.
* **Linear Models (Logistic Reg. & LDA):** ~56% Accuracy using the subset of best features (Lag1, Lag2).
* **KNN:** ~50-53% Accuracy. Proved less effective than discriminant analysis methods for this specific dataset.

## Technologies
* Python
* Pandas (Data Processing)
* Scikit-Learn (Classification Algorithms & Metrics)
* Matplotlib & Seaborn (Heatmap & Visualization)
