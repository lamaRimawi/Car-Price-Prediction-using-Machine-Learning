# Car Price Prediction using Machine Learning

## Overview
This project implements various machine learning models to predict car prices using regression techniques. The project explores different algorithms including Linear Regression, Lasso, Ridge, Support Vector Regression (SVR), and Polynomial Regression, along with custom implementations of gradient descent algorithms.

## Team Members
- **Lama Khattib** - Student ID: 1213515
- **Nada Sameer** - Student ID: 1200202

## Dataset
The project uses a car dataset (`cars.csv`) containing information about various vehicles including:
- Car name and brand
- Engine specifications (capacity, cylinders, horsepower)
- Performance metrics (top speed)
- Physical attributes (number of seats)
- Price information in multiple currencies (SAR, AED, USD)

## Project Structure
```
├── cars.csv                 # Dataset file
├── ml2.py                   # Main Python implementation
├── secondML (1).pdf         # Detailed project report
└── README.md               # Project documentation
```

## Features Implemented

### 1. Data Preprocessing
- **Currency Conversion**: Standardizes prices from SAR and AED to USD
- **Missing Value Handling**: Fills missing numerical values with mean imputation
- **Outlier Removal**: Uses Interquartile Range (IQR) method to remove extreme values
- **Log Transformation**: Applied to target variable to stabilize variance
- **Feature Scaling**: MinMaxScaler for normalization
- **Categorical Encoding**: One-hot encoding for categorical variables

### 2. Machine Learning Models

#### Regression Models
- **Linear Regression**: Basic linear relationship modeling
- **Lasso Regression**: L1 regularization for feature selection
- **Ridge Regression**: L2 regularization with hyperparameter tuning
- **Support Vector Regression (SVR)**: RBF and Polynomial kernels
- **Polynomial Regression**: Degrees 2-10 with overfitting analysis

#### Custom Implementations
- **Closed-form Solution**: Direct computation using normal equation
- **Batch Gradient Descent**: Full dataset optimization
- **Stochastic Gradient Descent**: Single sample updates
- **Mini-batch Gradient Descent**: Small batch optimization

### 3. Model Selection & Feature Engineering
- **Forward Selection**: Iterative feature selection based on MSE improvement
- **Cross-validation**: Grid search for hyperparameter optimization
- **Performance Comparison**: RMSE-based model selection

## Installation & Requirements

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Required Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- math
- random

## Usage

### Running the Complete Analysis
```python
python ml2.py
```

### Key Functions

#### Data Preprocessing
```python
# Convert prices to USD
data['price_usd'] = data['price'].apply(convert_to_usd)

# Remove outliers
for col in ['price_usd', 'engine_capacity', 'horse_power']:
    data = remove_outliers(data, col)
```

#### Model Training and Evaluation
```python
# Train models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Evaluate performance
results = evaluate_model(model, X_val, y_val, "Model Name")
```

#### Custom Gradient Descent
```python
# Batch Gradient Descent
cost_batch, theta_batch, mse_batch = batch_gradient_descent(
    learning_rate=0.01, X=X_train_, y=y_train, epochs=1000
)
```

## Results Summary

### Best Performing Models
1. **Support Vector Regression (SVR)**
   - R²: 0.7866
   - RMSE: 0.2983
   - Best overall performance

2. **Ridge Regression**
   - Good balance between bias and variance
   - Effective regularization

3. **Polynomial Regression (Degree 2-3)**
   - Good fit without overfitting
   - Higher degrees show severe overfitting

### Key Findings
- **Feature Importance**: `horse_power`, `top_speed`, `engine_capacity`, `seats`, `cylinder`
- **Overfitting**: Polynomial degrees >3 show severe overfitting
- **Gradient Descent**: All implementations converge to similar solutions
- **Regularization**: Ridge regression provides best bias-variance tradeoff

## Visualizations
The project includes comprehensive visualizations:
- Prediction vs Actual scatter plots
- Residual analysis
- Learning curves for gradient descent
- Feature selection progression
- Model performance comparisons

## Performance Metrics
- **R² Score**: Coefficient of determination
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Root Mean Squared Error (RMSE)**: Square root of MSE

## Model Comparison Results

| Model | R² | MAE | MSE | RMSE |
|-------|----|----|-----|------|
| Linear Regression | 0.7795 | 0.2286 | 0.0919 | 0.3032 |
| Lasso Regression | 0.7811 | 0.2247 | 0.0913 | 0.3021 |
| Ridge Regression | 0.7866 | 0.2249 | 0.0890 | 0.2983 |
| **SVR (Best)** | **0.7866** | **0.2249** | **0.0890** | **0.2983** |

## Future Improvements
- Implement ensemble methods (Random Forest, Gradient Boosting)
- Add more sophisticated feature engineering
- Explore deep learning approaches
- Implement automated hyperparameter optimization
- Add model interpretability analysis

## License
This project is for educational purposes as part of a Machine Learning course assignment.

## Acknowledgments
- **Instructor**: Dr. Ismail Khater
- **Course**: Machine Learning - Section 2
- **Institution**: Faculty of Engineering and Technology, Electrical and Computer Engineering Department

## Repository
🔗 **GitHub Repository**: [Car-Price-Prediction-using-Machine-Learning](https://github.com/lamaRimawi/Car-Price-Prediction-using-Machine-Learning)

## Contact
For questions or collaboration:
- **Lama Khattib** (GitHub: [@lamaRimawi](https://github.com/lamaRimawi))
- **Nada Sameer**

## How to Clone and Run
```bash
# Clone the repository
git clone https://github.com/lamaRimawi/Car-Price-Prediction-using-Machine-Learning.git

# Navigate to project directory
cd Car-Price-Prediction-using-Machine-Learning

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the main script
python ml2.py
```

---

**Note**: This project demonstrates various machine learning techniques for regression problems, with emphasis on proper data preprocessing, model selection, and performance evaluation.
