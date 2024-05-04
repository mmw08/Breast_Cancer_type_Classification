# Breast_Cancer_type_Classification

Breast cancer presents a significant public health challenge worldwide. This project endeavors to refine the classification of breast cancer types by harnessing biological data to identify the most effective predictor of cancer type. Various advanced machine learning models, including logistic regression, SVM, random forest, decision trees, gradient boosting, XGBoost, LightGBM, k-nearest neighbors, Naïve Bayes, and neural networks, were employed. Special attention was given to XGBoost due to its efficiency in handling large datasets which are imbalanced with high-dimensional features.

## Overview

### Data
- **data.csv**: Dataset containing features related to breast cancer diagnosis and corresponding labels indicating cancer types.

### Notebooks
1. **main.ipynb**: Jupyter Notebook detailing the methodology, data preprocessing, model training, and evaluation using various machine learning approaches. Finally, it contains the best model with hight F1 Score and accuracy Value.
2. **best_model.ipynb**: Jupyter Notebook containing the trained XGBoost model optimized for the oversampling approach and preprocessing steps for test data that will be provided by the user.

## Methodology

### Approach 1: Reference Approach
- Utilized various machine learning models for classification.
- Models were trained and fine-tuned to determine the best performer.
- XGBoost emerged as the leading candidate with an accuracy of 0.81 and F1 score of 0.73.

### Approach 2: PCA
- Applied Principal Component Analysis (PCA) to reduce feature dimensionality.
- Models trained on reduced feature set to assess impact on accuracy and F1 score.
- PCA effectively condensed features without significantly affecting model performance.

### Approach 3: Oversampling Data
- Employed SMOTE technique to address class imbalance.
- Enhanced number of minority class samples to improve model performance.
- XGBoost and Gradient Boost models demonstrated improved performance.

### Approach 4: Balanced Sampled Data
- Applied sample weighting mechanism to handle class imbalance.
- Gradient Boost model exhibited notable improvement in accuracy and F1 score.

### Approach 5: Manual Feature Selection
- Selected top 10 relevant features based on research and analysis.
- Models trained on reduced feature set to assess impact on performance.
- Feature selection did not significantly improve models' F1 scores.

### Approach 6: Feature Selection Model
- Incorporated model-based feature selection using Random Forest Classifier.
- Minimal impact observed on model performance compared to manual selection.
- XGBoost demonstrated slight improvement in accuracy and F1 score.

## Best Model Evaluation

- Tested models with highest accuracy and F1 scores on unseen test dataset.
- XGBoost with oversampling achieved highest accuracy (0.8) and F1 score (0.74) on the test dataset.
- XGBoost's stability across different datasets underscores its effectiveness in managing class imbalance.

## Conclusion

- XGBoost, enhanced with oversampling, emerged as the most effective model for breast cancer classification.
- Methodical data preparation and model tuning played crucial roles in achieving high performance.


This study highlights the potential of machine learning in improving breast cancer classification and underscores the importance of rigorous data analysis and model selection in predictive analytics.
