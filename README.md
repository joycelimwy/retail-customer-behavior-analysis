# Retail Customer Behavior Analysis

Analyze retail customer purchasing behavior and explore machine learning models to predict future coupon redemption using transaction data, exploratory data analysis, and customer spending patterns.

## Project Overview

This project analyzes retail transaction data to explore customer purchasing behavior and promotional activity. The analysis focuses on household spending patterns, shopping frequency, product departments, and coupon redemption.

The predictive modeling task is to estimate whether a household will redeem a coupon after Day 500 using purchasing behavior observed on or before Day 500.

## Objectives

- Explore customer spending and shopping frequency.
- Examine differences between households with and without recorded coupon redemptions.
- Investigate purchasing patterns across product departments.
- Create household-level features for predictive modeling.
- Establish a baseline model and compare Logistic Regression models.
- Evaluate model performance using appropriate classification metrics.
- Explore prediction thresholds and identify potential business applications.

## Dataset

The project uses retail transaction and supporting datasets, including:

- **Transaction data:** Household purchases, basket IDs, product IDs, quantities, sales values, store IDs, discounts, and transaction timing.
- **Product data:** Product departments, categories, brands, manufacturers, and product sizes.
- **Household demographic data:** Household characteristics such as age group, income, marital status, and household size.
- **Coupon redemption data:** Household coupon redemption records.
- **Coupon and campaign data:** Coupon-to-product mappings and campaign information.
- **Causal data:** Product, store, week, display, and mailer information.

The transaction dataset contained 2,595,732 rows before cleaning. Records with the product subcategory `GASOLINE-REG UNLEADED` were excluded from the transaction analysis.

## Methodology

### 1. Data Cleaning and Preparation

- Removed exact duplicate rows from the coupon dataset.
- Excluded gasoline transactions identified by the `GASOLINE-REG UNLEADED` product subcategory.
- Merged transaction data with product and household demographic information using left joins.
- Checked key uniqueness and merge relationships.
- Retained transactions without matching demographic information rather than dropping them. Demographic analysis was limited to households with available demographic records.

After gasoline transactions were excluded and the data was merged, the combined transaction dataset contained 2,570,770 rows and 26 columns.

### 2. Exploratory Data Analysis (EDA)

The exploratory analysis examined:

- Total sales by product department.
- The distribution of transaction-level sales values.
- Total household spending.
- Household shopping frequency.
- The relationship between shopping frequency and total spending.
- Differences in spending and shopping frequency between households with and without recorded coupon redemptions.
- Coupon redemption patterns across household income groups.

Potential outliers in sales value were reviewed. High-value purchases were not automatically removed because some were legitimate products. Quantity outlier detection using the IQR rule was not useful because the first and third quartiles were both 1.

### 3. Feature Engineering

Household-level features were created, including:

- **Total Spending:** Total sales value for each household.
- **Shopping Frequency:** Number of unique baskets for each household.
- **Average Basket Value:** Total spending divided by shopping frequency.
- **Average Quantity per Basket:** Total quantity divided by shopping frequency.
- **Department Spending:** Household spending aggregated by product department.

### 4. Predictive Modeling

The prediction task was defined as predicting whether a household would redeem at least one coupon after Day 500.

- **Feature period:** Transactions on or before Day 500.
- **Target period:** Coupon redemptions after Day 500.
- **Target variable:** 1 if a household had at least one recorded coupon redemption after Day 500; otherwise 0.
- **Train-test split:** 80% training and 20% testing, using stratification and `random_state=42`.
- **Baseline model:** `DummyClassifier(strategy='most_frequent')`.
- **Classification models:** Standard Logistic Regression and class-weighted Logistic Regression.
- **Cross-validation:** Five-fold stratified cross-validation.
- **Hyperparameter tuning:** `GridSearchCV` compared regularization strengths and class-weighting options.
- **Threshold analysis:** Several prediction thresholds were compared using out-of-fold probabilities generated from the training data.

The held-out test set was used for final model evaluation.

## Results

### 1. Exploratory Findings

- Household spending was right-skewed, with a relatively small number of households accounting for substantially higher spending.
- Shopping frequency and total household spending had a positive correlation of approximately **0.651**.
- Households with recorded coupon redemptions had higher average and median spending and higher shopping frequency than households without recorded redemptions.
- Grocery was the largest department by total sales.
- Demographic information was available for only 801 of the 2,500 households in the transaction data. Demographic findings therefore represent only households with available demographic records.

These are descriptive associations and do not establish that coupon redemption causes higher spending.

### 2. Baseline Model

The baseline `DummyClassifier` achieved **86.2% accuracy** on the test set. However, it predicted that every household would not redeem a coupon.

| Metric | No redemption (Class 0) | Redemption (Class 1) |
| :--- | :---: | :---: |
| Precision | 0.86 | 0.00 |
| Recall | 1.00 | 0.00 |
| F1-score | 0.93 | 0.00 |
| Support | 431 | 69 |

The baseline's accuracy was driven by the majority class. It did not identify any households that redeemed coupons, demonstrating why accuracy alone is insufficient for evaluating this prediction task.

### 3. Logistic Regression Models

The standard Logistic Regression model achieved **83.8% accuracy**, but its recall for coupon redeemers was only **0.07**.

Class weighting improved the model's ability to identify coupon redeemers. At the default prediction threshold of 0.5, the balanced Logistic Regression model achieved:

- **Accuracy:** 78.0%
- **Precision for coupon redeemers:** 0.34
- **Recall for coupon redeemers:** 0.59
- **F1-score for coupon redeemers:** 0.43
- **ROC-AUC:** 0.811

Although the balanced model had lower overall accuracy than the baseline, it identified coupon redeemers that the baseline missed entirely.

### 4. Cross-Validation and Hyperparameter Tuning

Five-fold stratified cross-validation of the balanced Logistic Regression model produced a mean ROC-AUC of approximately **0.810**, with a standard deviation of **0.012**. Its mean F1-score was approximately **0.465**.

Grid Search tested regularization strengths (`C`) of 0.01, 0.1, 1, 10, and 100, with and without balanced class weights.

The best configuration was:

- **Regularization parameter (`C`):** 100
- **Class weight:** `balanced`
- **Mean cross-validation F1-score:** 0.466

All tested configurations using balanced class weights outperformed the unweighted configurations on F1-score. However, several balanced configurations had very similar results, so the improvement from selecting `C = 100` over other balanced settings was small.

### 5. Prediction Threshold Comparison

Prediction thresholds were compared using out-of-fold probabilities generated through five-fold cross-validation on the training data.

| Threshold | Precision | Recall | F1-score |
| :---: | :---: | :---: | :---: |
| 0.3 | 0.262 | 0.870 | 0.403 |
| 0.4 | 0.306 | 0.764 | 0.437 |
| 0.5 | 0.358 | 0.667 | 0.466 |
| 0.6 | 0.402 | 0.562 | 0.468 |
| 0.7 | 0.427 | 0.453 | 0.439 |

A threshold of **0.6** achieved the highest F1-score among the tested thresholds, narrowly exceeding the score at 0.5. A threshold of 0.3 achieved the highest recall, while 0.7 achieved the highest precision.

This demonstrates a trade-off between identifying more potential coupon redeemers and reducing false-positive predictions.

### 6. Final Model Evaluation

The tuned Logistic Regression model was evaluated on the held-out test set using the selected threshold of **0.6**.

| Metric | Result |
| :--- | :---: |
| Accuracy | 0.812 |
| Precision for coupon redeemers | 0.368 |
| Recall for coupon redeemers | 0.507 |
| F1-score for coupon redeemers | 0.427 |
| ROC-AUC | 0.801 |

The confusion matrix showed that:

- **35 true positives:** Households that redeemed coupons and were correctly identified.
- **60 false positives:** Households predicted to redeem coupons that did not.
- **34 false negatives:** Households that redeemed coupons but were missed by the model.
- **371 true negatives:** Households that did not redeem coupons and were correctly identified.

The model correctly identified 35 of the 69 actual coupon redeemers. At the selected threshold, precision and accuracy were higher than at the default threshold, while recall and F1-score were slightly lower.

The model demonstrates useful predictive signal, but false positives and missed redeemers remain important limitations.

### 7. Feature Coefficient Analysis

In the balanced Logistic Regression model, the features with the largest positive coefficients were:

- Grocery spending.
- Average quantity per basket.
- Total spending.
- Produce spending.

The features with the largest negative coefficients included:

- Average basket value.
- `CNTRL/STORE SUP`.
- `HBC`.
- `MEAT-PCKGD`.

These coefficients indicate associations with predicted coupon redemption in the fitted model. They do not establish that a particular purchasing behavior causes coupon redemption. Correlated features may also influence one another's coefficients.

## Business Recommendations

### 1. Improve Customer Targeting

Shopping frequency, grocery spending, and average quantity per basket may be useful signals for identifying households for further promotional targeting experiments. The model should be validated on new data before being used in a live campaign.

### 2. Test Promotional Effectiveness

Retailers should conduct controlled experiments to determine whether coupons increase incremental spending or purchase frequency. This would help distinguish the effect of promotions from the existing spending habits of customers who already redeem coupons.

### 3. Balance Reach and Targeting Costs

The prediction threshold should reflect the retailer's business objective. A lower threshold may identify more potential coupon redeemers but generate more false positives. A higher threshold may reduce unnecessary targeting but miss more potential redeemers.

### 4. Improve Future Models

Future work could incorporate historical coupon redemption before the prediction cutoff, evaluate additional customer features, compare alternative classification algorithms, and validate performance on new or later-period data before deployment.

## Limitations

- **Incomplete demographic data:** Demographic information was available for only 801 of the 2,500 households, limiting demographic analysis.
- **Class imbalance:** Only 345 households redeemed a coupon after Day 500, compared with 2,155 that did not.
- **False-positive predictions:** At the selected threshold, the model incorrectly identified 60 non-redeeming households as potential redeemers.
- **Missed redeemers:** The model failed to identify 34 of the 69 actual coupon redeemers in the test set.
- **Observational data:** Associations between spending, shopping behavior, and coupon redemption do not establish causal effects.
- **Threshold sensitivity:** Precision and recall change with the prediction threshold. The selected threshold is a candidate based on the tested criteria, not proof of business effectiveness.
- **Selected prediction period:** The task uses Day 500 as a cutoff; results may differ with another observation period.

## Tools and Technologies

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Repository Contents

The repository contains the Jupyter Notebook(s) used for data cleaning, exploratory data analysis, feature engineering, predictive modeling, and evaluation, along with this README.
