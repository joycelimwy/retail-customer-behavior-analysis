# retail-customer-behavior-analysis
Analyze retail customer purchasing behavior and explore machine learning models to predict future coupon redemption using transaction data, exploratory data analysis, and customer spending patterns.

## Project Overview

This project analyzes retail transaction data to explore customer purchasing behavior and promotional activity. The analysis focuses on household spending patterns, shopping frequency, product departments, and coupon redemption.

The project also establishes a baseline machine learning model to predict future coupon redemption using customer purchasing behavior.

## Objectives

- Explore customer spending and shopping frequency.
- Examine differences between households that redeemed coupons and those that did not.
- Investigate purchasing patterns across product departments.
- Prepare household-level features for predictive modeling.
- Establish a baseline model for predicting future coupon redemption.

## Dataset

The project uses retail transaction and supporting datasets, including:

- Transaction data: Household purchases, basket IDs, product IDs, quantities, sales values, store IDs, discounts, and transaction timing.
- Product data: Product departments, categories, brands, manufacturers, and product sizes.
- Household demographic data: Household characteristics such as age group, income, marital status, and household size.
- Coupon redemption data: Household coupon redemption records.
- Coupon and campaign data: Coupon-to-product mappings and campaign information.
- Causal data: Product, store, week, display, and mailer information.

The transaction data contains 2,595,732 rows before cleaning. 
The analysis excludes records whose product subcategory is GASOLINE-REG UNLEADED.

## Methodology

### 1. Data Cleaning and Preparation

- Removed exact duplicate rows from the coupon dataset.
- Excluded gasoline transactions identified by the GASOLINE-REG UNLEADED product subcategory.
- Merged transaction data with product and household demographic information using left joins.
- Checked key uniqueness and merge relationships.
- Retained transactions without matching demographic information rather than dropping them. Demographic analysis was limited to households with available demographic records.

After removing gasoline transactions and merging the data, the combined transaction dataset contained 2,570,770 rows and 26 columns.

### 2. Exploratory Data Analysis

The exploratory analysis examined:

- Total sales by product department.
- The distribution of transaction-level sales values.
- Total household spending.
- Household shopping frequency.
- The relationship between shopping frequency and total spending.
- Coupon redemption rates and spending differences between redeemers
and non-redeemers.
- Coupon redemption patterns across household income groups.

Potential outliers in sales value were reviewed. High-value purchases were not automatically removed because some were legitimate products. 
Quantity outlier detection using the IQR rule was not useful because the first and third quartiles were both 1.

### 3. Feature Engineering

Household-level features were created, including:

- Total Spending: Total sales value for each household.
- Shopping Frequency: Number of unique baskets for each household.
- Average Basket Value: Total spending divided by shopping frequency.
- Average Quantity per Basket: Total quantity divided by shopping frequency.
- Department Spending: Household spending aggregated by product department.

### 4. Predictive Modeling

The prediction task was defined as predicting whether a household would
redeem a coupon after Day 500.

- Feature period: Transactions on or before Day 500.
- Target period: Coupon redemptions after Day 500.
- Target: Whether each household had at least one coupon redemption after Day 500.
- Train-test split: 80% training and 20% testing, using stratification and random_state=42.

A DummyClassifier with the most_frequent strategy was used as the baseline model. 
It predicts the majority class for every household.

## Results

### Exploratory Findings

- Household spending was right-skewed, with a relatively small number of households accounting for substantially higher spending.
- Shopping frequency and total household spending had a positive correlation of approximately 0.651.
- Households with coupon redemptions had higher average and median spending and higher shopping frequency than households without recorded redemptions.
- Grocery was the largest department by total sales.
- Demographic information was available for only 801 of the 2,500
households in the transaction data. Demographic findings therefore
represent only the households with available demographic records.

These are descriptive associations and do not establish that coupon redemption causes higher spending.

### Baseline Model

The baseline DummyClassifier achieved 86.2% accuracy on the test set. However, it predicted that every household would not redeem a coupon.

| Metric | No redemption (Class 0) | Redemption (Class 1) |
| :--- | :---: | :---: |
| **Precision** | 0.86 | 0.00 |
| **Recall** | 1.00 | 0.00 |
| **F1-score** | 0.93 | 0.00 |
| **Support** | 431 | 69 |

The baseline’s accuracy is driven by the majority class. It does not identify any households that redeem coupons, so accuracy alone is not sufficient to evaluate the usefulness of a future predictive model.

## Limitations

- Demographic data is available for only a subset of households, limiting demographic analysis.
- The target is imbalanced: fewer households redeemed coupons after Day 500 than did not.
- The baseline model predicts only the majority class and does not identify future coupon redeemers.
- The analysis is observational. Associations between spending, shopping behavior, and coupon redemption should not be interpreted as causal effects.
- The prediction task uses a selected cutoff at Day 500; results may differ with another cutoff or observation period.

## Tools and Technologies

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- scikit-learn

### Repository Contents

The repository contains the Jupyter Notebook(s) used for data cleaning, exploratory data analysis, feature engineering, and baseline modeling, along with this README.
