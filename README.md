# risky_bets

*m148 final project*

  
  

### i. Description of Dataset

**Dataset name:** Diabetes Health Indicators Dataset  

**Source:** Kaggle [link](https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset/data)  

**Rows:** 100,000 patients  

**Columns:** ~31 features  

**File:** 'diabetes_data.csv'  

  

**Brief summary:**  

This dataset contains 100,000 synthetic but clinically realistic patient profiles modeled after survey patterns from the CDC and medical research. It includes demographic, lifestyle, biometric, and laboratory features used to predict diabetes status, with no missing values and a moderately balanced target distribution. Because many indicators resemble real-world risk factors (such as blood pressure, cholesterol, physical activity, BMI, and education), the dataset is well suited for our health-focused machine learning model's predictive accuracy and interpretability.

  
  

---

### ii. Project Idea

For our final project, we will compare a variety of supervised and unsupervised machine learning methods for diabetes prediction. We address two problems that are essential applications of the Data Science Life Cycle:

1. Do neural networks outperform traditional machine learning models in predicting diabetes?

2. How do different data preprocessing and feature engineering methods affect diabetes prediction accuracy?

These questions allow us to compare classical applied statistics vs. modern ML approaches while also evaluating how data preparation influences performance.

  
  

---

### iii. Key Methodology

  
  
  
  

---

### iv. Results

  
  
  
  

---

### v. How To Run the Code

  
  
  
  

---

---

## ***Appendix***

  

### i. Explain the exploratory data analysis that you conducted. What was done to visualize your data and split your data for training and testing?

First, apply info function to request general information about the dataset.

<div align="center">
<img src="md_pictures/eda_output.png">
</div>


There are 31 columns and 100000 rows, which means that there are 100000 individuals and
30 predictors after excluding `diagnoseddiabetes` column. All 31 columns are complete, with
no missing values, and each column contains exactly 100000 non-null entries. The entries
have `int`, `object`, `float` types. Therefore, it is necessary to apply on-hot encoding to transform
categorical data into numerical data. This structure provides a clean and comprehensive basis
for downstream analysis and modeling.
  
Second, since some of the datatypes are object, it is necessary to perform on-hot encoding. Step
one is to delete individuals who answer ”other” in Gender and Ethnicity. The next step is to
divide answers into different predictors.

Apply describe function to request numerical details of each predictors.

<div align="center">

<strong>Table 1: Summary Statistics of All Variables (Numerical + One-hot)</strong>

<img src="md_pictures/Table_1.png">

</div>

Observing the distribution of the target variable, it is found that nearly 60% of the individuals
are diagnosed as diabetes. The target variable `Diabetes` is highly balanced, which is good for
model training and evaluation. There is no obvious majority of non-diabetes and diabetes. If a
model only considers the majority class, it might always predict 0 (non-diabetic), but failing to
capture true positive cases, making it practically useless for prediction. Metrics like precision,
recall, F1-score, and ROC-AUC are more appropriate to judge the accuracy of the models based
on this dataset.

Third, investigating collinearity of the predictors in order to delete super similar predictors. The
following graph is the visualization of collinearity matrix:

<div align="center">
<img src="md_pictures/collinearity_matrix.png">
</div>


Select predictors that are highly correlated with a threshold over 0.9 in absolute value. Predictors such as `ldl_cholesterol`, `hba1c`, and `diabetes_stage_Type 2` are deleted in order to
simplify the data.


<div align="center">

<strong>Data Visualizations</strong>

</div>

<div align="center">
<img src="md_pictures/fig1.png">
</div>
<div align="center">

Figure 1: Left: Gender Count. Right: Diabetes Count by Gender.

</div>

<div align="center">
<img src="md_pictures/fig2.png">
</div>
<div align="center">

Figure 2: Left: BMI vs Diabetes Risk Score. Right: Diet Score by Diabetes Status.

</div>

<div align="center">
<img src="md_pictures/fig3.png">
</div>
<div align="center">

Figure 3: Left: Age Distribution by Diabetes Status. Right: Fasting vs Postprandial Glucose
by Gender.

</div>

After these EDA processes, we split the dataset randomly into 80% training dataset and 20%
testing dataset. Their names are `X_train` for the independent variable training set, `X_test`
for the independent variable testing set, `y_train` for the dependent variable training set, and
`y_test` for the dependent variable testing set




### ii. What data pre-processing and feature engineering (or data augmentation) did you complete on your project?

In EDA analysis section, one-hot encoding and finding collinearity have been done. Them,
dealing with outliers and preparing normalized data.
  




### iii. How was regression analysis applied in your project? What did you learn about your data set from this analysis and were you able to use this analysis for feature importance? Was regularization needed?

Predicting whether an individual has diabetes is a binary judgement, which means a diabetes
score parameter is needed to imply how likely it is for an individual to has diabetes. And the
threshold is set to be 0.5. Since there are over 30 predictors for regression, it was decided that
Lasso Regression is the main method performed for Regression Analysis because it eliminates
unimportant predictors by punishing their coefficients into 0.

The dataset was first split into training (80%) and testing (20%) sets, and all predictors were
normalized to ensure comparability of coefficients. A 5-fold cross-validation procedure was
applied to select the optimal regularization parameter α and to prevent overfitting. The Lasso
regression model can be expressed as:

$$
\hat{\beta} = \arg\min_{\beta} \left( 
  \frac{1}{2n} \sum_{j=1}^{n} \left( y_j - \beta_0 - \sum_{i=1}^{p} \beta_i X_{ij} \right)^2 + \lambda \sum_{i=1}^{p} |\beta_i|
\right)
$$

$$
\text{diabetes\_score} = \beta_0 + \sum_{i=1}^{p} \beta_i X_i
$$
where $\beta_i$ represents the coefficient for predictor $X_i$.

  
The coefficients of the well-performed model are:

<div align="center">

<strong>Table 2: Lasso Regression Coefficients for Diabetes Prediction</strong>

<img src="md_pictures/Table_2.png">

</div>

Other features, such as age, BMI, waist-to-hip ratio, and physical activity, had smaller but
nonzero coefficients. Many categorical variables (e.g., education level, employment status,
ethnicity) were shrunk to zero by Lasso, indicating lower predictive importance.

The best α found by cross-validation was `0.000655`, and the cross-validated `R^2` score on
the training set was approximately `0.822`. It seems that Regulation is needed for the regression
prediction.



### iv. How was logistic regression analysis applied in your project? What did you learn about your data set from this analysis and were you able to use this analysis for feature importance? Was regularization needed?

Predicting whether an individual has diabetes is a binary classification problem. A logistic regression model was applied to estimate the probability of diabetes for each individual. The
threshold for classification was set at 0.5. Since there are over 30 predictors, L1-regularized
logistic regression (Lasso) was chosen to perform feature selection by shrinking unimportant
coefficients to zero.

The dataset was split into training (80%) and testing (20%) sets, and all predictors were standardized to have zero mean and unit variance. A 5-fold cross-validation procedure was applied
to select the optimal inverse regularization strength `C` and to prevent overfitting. The logistic
regression model can be expressed as:

The logistic regression model can be expressed as:

$$
P(Y = 1 \mid X) = \frac{1}{1 + \exp\!\left( -\left( \beta_0 + \sum_{i=1}^{p} \beta_i X_i \right) \right)}
$$

where $\beta_i$ represents the coefficient for predictor $X_i$, and $P(Y = 1 \mid X)$ is the probability that an individual has diabetes.

  
The coefficients of the well-performed model are:

<div align="center">

<strong>Table 3: Logistic Regression Coefficients for Diabetes Prediction</strong>

<img src="md_pictures/Table_3.png">

</div>


Many predictors, such as physical activity, diet score, family history of diabetes, and categorical variables like education level, ethnicity, and employment status, were shrunk to zero by L1
regularization, indicating low predictive importance.

The best `C` (inverse of regularization strength) found by cross-validation was approximately
`0.046`, and the training accuracy was `0.969`. The model achieved a test accuracy of `0.968`. The
confusion matrix and classification report are shown below:

$$
\text{Confusion Matrix} =
\begin{bmatrix}
7171 & 288 \\
300  & 10850
\end{bmatrix}
$$

Classification report:
- **Class 0 (No Diabetes)**: precision = 0.96, recall = 0.96, F1-score = 0.96
- **Class 1 (Diabetes)**: precision = 0.97, recall = 0.97, F1-score = 0.97
- **Overall Accuracy**: 0.97










### v. How were KNN, decision trees, or random forest used for classification on your data? What method worked best for your data and why was it good for the problem you were addressing?

### vi. How were PCA and clustering applied on your data? What method worked best for your data and why was it good for the problem you were addressing?

### vii. Explain how your project attempted to use a neural network on the data and the results of that attempt.

### viii. Give examples of hyperparameter tuning that you applied in preparing your project and how you chose the best parameters for models.
