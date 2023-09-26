+++
date = '2021-01-20T06:09:46-04:00'
title = 'Linear Regression : A Comprehensive Overview '
#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++


## Introduction

Linear regression is a basic machine learning algorithm used for supervised learning tasks involving prediction and forecasting. It is used to model the relationship between a dependent variable (target variable) and one or more independent variables (features/predictors) by fitting a linear equation to the observed data.  

## Sample Input

The input to a linear regression model consists of:

- A dataset with one or more independent variables (X) and a continuous target variable (y) 
- The data is split into training and test sets
- The training data is used to train the model i.e. fit the regression line
- The test data is used to evaluate model performance

Sample input data:

| X1 | X2 | y |  
|-|-|-|
| 0.5 | 2.1 | 1.1 |
| 1.3 | 0.7 | 2.8 | 
| ... | ... | ... |

Where X1 and X2 are independent variables, y is the target variable.

## Sample Output

The output from a linear regression model is:

- Regression coefficients (intercept and slopes) that define the regression line 
- Predicted values of the target variable for given inputs
- Performance metrics like R-squared, Mean Squared Error etc. on training and test data

Sample output:

Intercept (β0): 1.5 
Slope (β1): 0.3
Slope (β2): 0.8

R-squared: 0.75 
MSE: 0.05

## Target Variable

Linear regression models a continuous target variable based on its linear relationship with the predictors. The target variable must be numeric and unbounded.

Common use cases are predicting home prices, stock prices, sales revenue etc based on factors like size, location, marketing budget etc.

## Problems Algorithm Handles

Linear regression is used for:

- Predictive modeling - Predicting a numeric target value based on known values of predictors 
- Forecasting - Projecting future values of time series data based on historical patterns
- Exploratory data analysis - Determining relationship between dependent and independent variables  

It handles regression problems i.e. supervised learning problems involving prediction of a numeric value. It cannot handle classification problems with categorical target variables.

## Datasets

Some commonly used datasets for linear regression are:

- Housing Data: Predicting house prices based on features like area, bedrooms etc.
- Stock Price Data: Modeling stock prices using factors like market cap, PE ratio etc.  
- Student Marks Data: Predicting student exam scores based on studying hours, tuition etc.
- Car Price Data: Estimating car prices using attributes like mileage, age etc.

These are available on Kaggle and other open sources.

## Full Code Example

```python
# import libraries
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# load data
df = pd.read_csv('housing.csv')

# split data into X and y
X = df[['area', 'bedrooms', 'age']]  
y = df['price']

# split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train the model on training set 
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on test set
y_pred = model.predict(X_test)

# model evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(rmse)

# print coefficients
print(model.intercept_)
print(model.coef_)
```

This loads the data, trains a LinearRegression model, makes predictions on test data, and evaluates model performance.

## Math 

The linear regression model defines a linear relationship between the target variable y and predictors X:

y = β0 + β1X1 + β2X2 + ... + βnXn

Where,

β0 is the intercept 

β1 to βn are the regression coefficients for each predictor 

The coefficients are estimated using the least squares method, by minimizing the sum of squared residuals between actual and predicted values of y.

Various optimization algorithms like gradient descent can be used to optimize this loss function and learn the coefficients.

## Cost Function

The most commonly used cost function for linear regression is Mean Squared Error (MSE). It calculates the average squared difference between actual and predicted values.

MSE = 1/n Σ (yi - ŷi)2

Where yi is the actual value, ŷi is the predicted value and n is the number of samples.

Minimizing MSE via gradient descent gives the optimized regression coefficients. Other loss functions like MAE, Huber etc can also be used.

## Real-Time Applications

Some real-world applications of linear regression include:

- Predicting housing prices based on area, bedrooms, other factors
- Forecasting sales, revenue based on historical data, marketing spend etc  
- Estimating lifespan of equipment based on usage, maintenance etc. 
- Predicting stock price trends based on financial indicators
- Estimating length of stay for hospital patients based on age, condition etc.

It is easy to implement and fast to train, hence widely used for both simple and complex real-world predictions involving continuous numeric variables.

## Limitations

Some key limitations of linear regression:

- Assumes linear relationship between target and predictors - fails for complex nonlinear relationships
- Prone to overfitting with many input features
- Outliers can skew the fitted line
- Cannot directly model categorical variables
- Makes assumptions about underlying distributions of data
- Cannot handle non-numeric output variables   

> Due to these limitations, linear regression may not perform well on complex real-world data. Advanced algorithms like neural networks may be more suitable in such cases.

Here are the additional details for Linear Regression:

## Assumptions

The main assumptions made by linear regression are:

- Linear relationship - The target variable has a linear relationship with the predictors
- Multivariate normality - The residuals are normally distributed  
- No or little multicollinearity - The predictors are not highly correlated with each other
- Homoscedasticity - The variance of residuals is constant across data points
- Independence - The residuals are independent and identically distributed
- Lack of autocorrelation - The residuals are uncorrelated with each other 

These assumptions should hold true for linear regression to make reliable predictions. Violating them can result in suboptimal model performance.

## Things to Keep in Mind

Some things to keep in mind when applying linear regression:

- Check assumptions and transform data if needed 
- Remove outliers or use robust methods if outliers impact model
- Feature selection to remove irrelevant variables  
- Address multicollinearity if predictors are correlated
- Regularization if model is overfitting training data
- Timeseries data may need autocorrelation modeling  
- Assumptions may not hold true for complex real-world data
- Interpret coefficients carefully for correlations, not causations

## Model Evaluation Metrics 

Evaluation metrics for linear regression include:

- Mean Absolute Error (MAE) - Average absolute difference between predicted and actual values. Use for human interpretability.
- Mean Squared Error (MSE) - Average squared difference between predicted and actual values. Sensitive to outliers.
- Root Mean Squared Error (RMSE) - Square root of MSE. Used for scale-dependent problems.
- R-squared - Statistical measure representing variance explained by the model. Values closer to 1 are better.
- Adjusted R-squared - Adjusted for number of predictors in the model. Useful for model selection.

## Avoiding Overfitting

Overfitting can be avoided by:

- Removing unnecessary features using feature selection
- Regularization methods like L1 and L2 regularization
- Reducing model complexity - fewer features or polynomial terms
- Early stopping - stop training when validation error stops decreasing 
- Getting more data samples to increase generalization
- Dropout - randomly dropping input units during training
- Cross-validation - evaluate model on unseen data

The choice depends on the dataset and use case. Simpler models are less prone to overfitting than complex ones.

## Handling Multicollinearity

Methods for handling multicollinearity include:

- Variance Inflation Factor (VIF) to identify and remove correlated features
- Principal Component Analysis (PCA) to convert correlated features into orthogonal principal components
- Ridge (L2) regularization constrains coefficient magnitudes 
- Grouping correlated features into a single feature
- Getting more data samples to make model more robust 

VIF and regularization are commonly used. PCA also helps but loses interpretability of original features.

