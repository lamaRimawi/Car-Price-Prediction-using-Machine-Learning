#lama 1213515 , nada 1200202
import math
import random
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Seed for reproducibility
SEED = 42

# Load the dataset
file_path = 'cars.csv'
data = pd.read_csv(file_path)

# function to convert prices to USD
def convert_to_usd(price):
    conversion_rates = {'SAR': 0.27, 'AED': 0.27}  # Conversion rates
    try:
        if isinstance(price, float):
            return price
        elif pd.isna(price) or price in ["TBD", "N/A"]:
            return np.nan
        elif 'SAR' in price:
            return float(price.replace('SAR', '').replace(',', '').strip()) * conversion_rates['SAR']
        elif 'AED' in price:
            return float(price.replace('AED', '').replace(',', '').strip()) * conversion_rates['AED']
        else:
            return float(price.replace(',', '').strip())
    except ValueError:
        return np.nan

# Convert and clean the price column
data['price_usd'] = data['price'].apply(convert_to_usd)
data.drop(columns=['price'], inplace=True)

# Convert numeric columns to proper format and handle missing values
numeric_columns = ['engine_capacity', 'cylinder', 'horse_power', 'top_speed', 'seats']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Remove outliers using the 1.5x IQR rule
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]

for col in ['price_usd', 'engine_capacity', 'horse_power']:
    data = remove_outliers(data, col)

# Apply log transformation to stabilize variance in the target variable
data['price_usd'] = np.log1p(data['price_usd'])

# Normalization
scaler = MinMaxScaler()  # Use MinMaxScaler for normalization
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# One-hot encode categorical variables
categorical_columns = ['car name', 'brand', 'country']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['price_usd'])
y = data['price_usd']

# Ensure only numeric columns remain in X
X = X.select_dtypes(include=[np.number])

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

# Fill NaN values in the features (X)
X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_val.mean())
X_test = X_test.fillna(X_test.mean())

# Fill NaN values in the target (y) (though should be rare after log transformation)
y_train = y_train.fillna(y_train.mean())
y_val = y_val.fillna(y_val.mean())
y_test = y_test.fillna(y_test.mean())


#function to print regression metrics
def print_regress_metric(actual, prediction):
    r2 = r2_score(actual, prediction)
    mae = mean_absolute_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    rmse = math.sqrt(mse)
    print(f"R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

# function to evaluate and visualize models
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    metrics = print_regress_metric(y, y_pred)

    # Plot predictions vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{dataset_name} - Predictions vs Actual")
    plt.grid()
    plt.show()

    return metrics

# Train and evaluate models
results = {}

# Linear Regression
print("Linear Regression:")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
results["Linear Regression"] = evaluate_model(linear_model, X_val, y_val, "Linear Regression")

# Lasso Regression
print("Lasso Regression:")
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
results["Lasso Regression"] = evaluate_model(lasso_model, X_val, y_val, "Lasso Regression")

# Ridge Regression
print("Ridge Regression:")
ridge_model = Ridge()
param_grid = {"alpha": [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(ridge_model, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train)
best_ridge = grid_search.best_estimator_
results["Ridge Regression"] = evaluate_model(best_ridge, X_val, y_val, "Ridge Regression")

# SVR
print("Support Vector Regression (SVR):")
svr_model = SVR(kernel="rbf", C=10, epsilon=0.2)
svr_model.fit(X_train, y_train)
results["SVR"] = evaluate_model(svr_model, X_val, y_val, "SVR")

# Final Test Set Evaluation for Best Model
print("Final Test Evaluation (Best Model - Ridge Regression):")
results["Test (Ridge)"] = evaluate_model(best_ridge, X_test, y_test, "Ridge Regression - Test")

# Residual Analysis
def plot_residuals(model, X, y, title):
    y_pred = model.predict(X)
    residuals = y - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color="blue")
    plt.title(f"{title} - Residual Analysis")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

plot_residuals(best_ridge, X_val, y_val, "Ridge Regression")

# Add intercept term to X (the bias term)
X_ = np.c_[np.ones(X_train.shape[0]), X_train]

# Closed-form solution for Linear Regression (theta = (X'X)^(-1) * X'y)
# Compute the inverse of the matrix X_.T @ X_
X_T_X_inv = np.linalg.inv(np.dot(X_.T, X_))

# Compute the theta using the closed-form solution
theta_closed_form = np.dot(X_T_X_inv, np.dot(X_.T, y_train))

print(f'Closed-form solution weights: {theta_closed_form}')

# Make predictions using the closed-form solution
y_pred_closed_form = np.dot(X_, theta_closed_form)


def batch_gradient_descent(learning_rate, X, y, epochs: int, return_model_result: bool = True):
    # Initial outputs to track progress
    mse_ = []
    cost_ = []
    theta_ = []
    n = X.shape[0]  # Number of training samples
    theta = np.ones(X.shape[1])  # Initialize weights
    X_transpose = X.T  # Precompute transpose of X

    for i in range(epochs):
        # Hypothesis (Predictions)
        hypothesis = np.dot(X, theta)

        # Loss (difference between predicted and actual values)
        loss = hypothesis - y

        # Cost function (Mean Squared Error)
        J = np.sum(loss ** 2) / (2 * n)

        # Print the cost and theta every 100 iterations
        if i % 100 == 0:
            print(f'Iter {i} | Cost: {J:.4f} | Theta: {theta}')

        # Gradient computation
        gradient = np.dot(X_transpose, loss) / n

        # Update theta (weights)
        theta = theta - learning_rate * gradient

        # Store metrics
        y_pred = np.dot(X, theta)
        mse_.append(mean_squared_error(y, y_pred))  # MSE for this iteration
        cost_.append(J)  # Store the cost
        theta_.append(theta)  # Store the weights for each iteration

    # Final print statement at the end of training
    print(f'End Iteration: Cost: {J:.4f} | Theta: {theta}')

    # Final prediction and metrics
    y_pred = np.dot(X, theta)
    mse_.append(mean_squared_error(y, y_pred))  # Final MSE
    cost_.append(J)  # Final cost

    if return_model_result:
        print(f'Final Model Performance: MSE: {mse_[-1]:.4f}, Cost: {cost_[-1]:.4f}')

    return cost_, theta_, mse_

# Set parameters
learning_rate = 0.01
n_epochs = 1000



# Prepare the data (add a column of ones for the intercept term in X)
X_train_ = np.c_[np.ones(X_train.shape[0]), X_train]

# Run Batch Gradient Descent
cost_batch, theta_batch, mse_batch = batch_gradient_descent(learning_rate=learning_rate,
                                                            X=X_train_,
                                                            y=y_train,
                                                            epochs=n_epochs)

# Make predictions using Batch Gradient Descent
y_pred_batch_gd = np.dot(X_train_, theta_batch[-1])  # Use final theta

# Assuming cost_batch and mse_batch contain the cost and MSE for each epoch
# Update variable names for the plots
fig, axs = plt.subplots(2, sharex=True, figsize=(13, 8))
fig.suptitle('Batch GD Performance')

# Plot the cost for each epoch
axs[0].plot(cost_batch)  # Plot cost over epochs
axs[0].set_title('Cost Each Epoch')
axs[0].set(ylabel='Cost')

# Plot the MSE for each epoch
axs[1].plot(mse_batch)  # Plot MSE over epochs
axs[1].set_title('MSE Each Epoch')
axs[1].set(ylabel='MSE', xlabel='Epoch')

plt.show()

# Define the SGD function
def get_pred(X, theta):
    '''Use the weights to predict yhat'''
    return np.dot(X, theta)

def grad_loss(X, y, theta):
    '''
    Compute gradient using MSE
    '''
    y_pred = get_pred(X, theta)
    error = y_pred - y
    loss_gradient = (np.dot(np.transpose(X), error)) / len(X)
    return loss_gradient

def _iter(X, y, batch_size=1):
    n_observations = X.shape[0]
    idx = list(range(n_observations))
    random.shuffle(idx)
    for batch_id, i in enumerate(range(0, n_observations, batch_size)):
        _pos = np.array(idx[i: min(i + batch_size, n_observations)])
        yield batch_id, X.take(_pos, axis=0), y.take(_pos)

# Define the SGD regressor function
def _sgd_regressor(X, y, learning_rate, n_epochs, batch_size=1):
    mse_log = []  # List to store MSE values for each epoch
    theta_log = []  # List to store theta values for each epoch
    np.random.seed(42)  # Set random seed for reproducibility
    theta = np.random.rand(X.shape[1])  # Initialize theta randomly

    for epoch in range(n_epochs + 1):
        total_error = 0
        for batch_id, data, label in _iter(X, y, batch_size):
            grad_loss_ = grad_loss(data, label, theta)
            theta = theta - learning_rate * grad_loss_  # Update weights

        # Compute MSE after each epoch
        y_pred = get_pred(X, theta)
        mse = mean_squared_error(y, y_pred)  # Calculate MSE
        mse_log.append(mse)  # Log MSE value
        theta_log.append(theta)  # Log theta value

        if epoch % 100 == 0:
            print(f'Epoch: {epoch} | MSE: {mse}')

    return theta, theta_log, mse_log

# Run Stochastic Gradient Descent (SGD)
learning_rate = 0.01
n_epochs = 1000
X_train_ = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept term to X
theta_sgd, theta_log, mse_sgd = _sgd_regressor(X_train_, y_train, learning_rate, n_epochs)


# Plot the MSE for each epoch
fig, axs = plt.subplots(1, figsize=(10, 6))  # Single subplot for performance
fig.suptitle('Mini-batch GD Performance')

# Plot the full MSE log
axs.plot(mse_sgd)
axs.set_title('MSE Each Epoch')
axs.set(ylabel='MSE', xlabel='Epoch')
plt.show()

# Plot the first 10 epochs of MSE
fig, axs = plt.subplots(1, figsize=(10, 6))  # Single subplot for performance
fig.suptitle('Mini-batch GD Performance (First 10 Epochs)')

# Plot the first 10 MSE values
axs.plot(mse_sgd[:10])  # Slice the first 10 MSE values
axs.set_title('MSE for First 10 Epochs')
axs.set(ylabel='MSE', xlabel='Epoch')
plt.show()

# Make predictions using Stochastic Gradient Descent
y_pred_sgd = np.dot(X_train_, theta_sgd)

# Plot MSE over epochs
fig, axs = plt.subplots(figsize=(10, 6))  # Create a single subplot
axs.plot(mse_sgd)  # Plot the MSE values
axs.set_title('MSE each epoch (Stochastic Gradient Descent)')  # Set title
axs.set(ylabel='MSE', xlabel='Epoch')  # Set axis labels
fig.suptitle('Stochastic Gradient Descent Performance')  # Title for the entire figure

plt.show()

# Print final weights and evaluate the model
print(f'Final weights: {theta_sgd}')
print_regress_metric(y_train, y_pred_sgd)

# Mini-Batch Gradient Descent implementation
def _sgd_regressor_minibatch(X, y, learning_rate, n_epochs, batch_size):
    mse_log = []
    np.random.seed(42)
    theta = np.random.rand(X.shape[1])  # Initialize theta randomly

    for epoch in range(n_epochs):
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            gradient = np.dot(X_batch.T, np.dot(X_batch, theta) - y_batch) / batch_size
            theta -= learning_rate * gradient  # Update weights

        # Compute MSE after each epoch
        y_pred = np.dot(X, theta)
        mse_log.append(mean_squared_error(y, y_pred))

    return theta, mse_log

# Run Mini-Batch Gradient Descent
theta_minibatch, mse_minibatch = _sgd_regressor_minibatch(X_train_, y_train, learning_rate=0.01, n_epochs=1000, batch_size=50)

# Make predictions using Mini-Batch Gradient Descent
y_pred_minibatch = np.dot(X_train_, theta_minibatch)

# Plot predictions for each method
plt.figure(figsize=(14, 8))

# Plot actual vs predicted for Closed-form
plt.subplot(2, 2, 1)
plt.scatter(y_train, y_pred_closed_form, color='blue', alpha=0.6, label='Closed-form Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', label='Perfect Prediction')
plt.title('Closed-form vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Plot actual vs predicted for Batch Gradient Descent
plt.subplot(2, 2, 2)
plt.scatter(y_train, y_pred_batch_gd, color='green', alpha=0.6, label='Batch GD Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', label='Perfect Prediction')
plt.title('Batch GD vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Plot actual vs predicted for Stochastic Gradient Descent
plt.subplot(2, 2, 3)
plt.scatter(y_train, y_pred_sgd, color='purple', alpha=0.6, label='SGD Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', label='Perfect Prediction')
plt.title('SGD vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Plot actual vs predicted for Mini-Batch Gradient Descent
plt.subplot(2, 2, 4)
plt.scatter(y_train, y_pred_minibatch, color='orange', alpha=0.6, label='Mini-Batch GD Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', label='Perfect Prediction')
plt.title('Mini-Batch GD vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.tight_layout()
plt.show()

# Create subplots to compare Closed-form and Gradient Descent results
plt.figure(figsize=(14, 8))

# Plot actual vs predicted for Closed-form vs Batch Gradient Descent
plt.subplot(2, 2, 1)
plt.scatter(y_train, y_pred_closed_form, color='blue', alpha=0.6, label='Closed-form Predictions')
plt.scatter(y_train, y_pred_batch_gd, color='green', alpha=0.6, label='Batch GD Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', label='Perfect Prediction')
plt.title('Closed-form vs Batch GD')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Plot actual vs predicted for Closed-form vs Stochastic Gradient Descent
plt.subplot(2, 2, 2)
plt.scatter(y_train, y_pred_closed_form, color='blue', alpha=0.6, label='Closed-form Predictions')
plt.scatter(y_train, y_pred_sgd, color='purple', alpha=0.6, label='SGD Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', label='Perfect Prediction')
plt.title('Closed-form vs SGD')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# Plot actual vs predicted for Closed-form vs Mini-Batch Gradient Descent
plt.subplot(2, 2, 3)
plt.scatter(y_train, y_pred_closed_form, color='blue', alpha=0.6, label='Closed-form Predictions')
plt.scatter(y_train, y_pred_minibatch, color='orange', alpha=0.6, label='Mini-Batch GD Predictions')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', label='Perfect Prediction')
plt.title('Closed-form vs Mini-Batch GD')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.tight_layout()
plt.show()


# Polynomial Regression with varying degrees (2 to 10)
def polynomial_regression(X_train, y_train, X_val, y_val):
    poly_results = {}
    for degree in range(2, 11):
        poly = PolynomialFeatures(degree=degree)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_val = poly.transform(X_val)

        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        y_pred = model.predict(X_poly_val)

        metrics = print_regress_metric(y_val, y_pred)
        poly_results[degree] = metrics

        # Plot predictions vs actual values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_val, y_pred, alpha=0.6, edgecolor="k")
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linewidth=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Polynomial Regression (Degree {degree}) - Predictions vs Actual")
        plt.grid()
        plt.show()

    return poly_results

# Support Vector Regression with GridSearchCV
def svr_with_grid_search(X_train, y_train, X_test, y_test, kernel="rbf"):
    # Reduced parameter grid for Polynomial Kernel
    param_grid = {
        'C': [0.1, 1, 10],  # Reduced values
        'epsilon': [0.1, 0.2],  # Practical epsilon values
        'degree': [2, 3] if kernel == "poly" else [3]  # Smaller degree for Polynomial Kernel
    }

    # Initialize SVR model
    svr_model = SVR(kernel=kernel)

    # GridSearchCV setup with verbosity and error handling
    grid_search = GridSearchCV(
        svr_model,
        param_grid,
        cv=6,
        n_jobs=-1,
        return_train_score=True,
        verbose=2,
        error_score="raise",
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Extract best parameters and test performance
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    y_pred = grid_search.predict(X_test)
    test_metrics = print_regress_metric(y_test, y_pred)

    # Print results
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation score: {best_score}")
    print(f"Test Metrics: {test_metrics}")

    # Visualization of predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"SVR ({kernel} Kernel) - Predictions vs Actual")
    plt.grid()
    plt.show()

    return best_params, test_metrics


# Call functions and evaluate models
poly_results = polynomial_regression(X_train, y_train, X_val, y_val)

print("Polynomial Regression Results:")
for degree, metrics in poly_results.items():
    print(f"Degree {degree}: {metrics}")

# SVR with RBF Kernel
print("SVR with RBF Kernel:")
svr_rbf_params, svr_rbf_metrics = svr_with_grid_search(X_train, y_train, X_test, y_test, kernel="rbf")

# SVR with Polynomial Kernel
print("SVR with Polynomial Kernel:")
svr_poly_params, svr_poly_metrics = svr_with_grid_search(X_train, y_train, X_test, y_test, kernel="poly")

# Summarized results
print("\nSummary of SVR Results:")
print(f"RBF Kernel - Best Parameters: {svr_rbf_params}, Metrics: {svr_rbf_metrics}")
print(f"Polynomial Kernel - Best Parameters: {svr_poly_params}, Metrics: {svr_poly_metrics}")

# Function to calculate and print regression metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}


# General model evaluation function
def evaluate_model(model, X, y, dataset_name="Dataset"):
    y_pred = model.predict(X)
    metrics = print_regress_metric(y, y_pred)

    # Plot predictions vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{dataset_name} - Predictions vs Actual")
    plt.grid()
    plt.show()

    print(f"\n{dataset_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics

def model_selection(models, X_train, y_train, X_val, y_val, metric='RMSE'):
    best_model = None
    best_metrics = None
    rmse_values = []  # To store RMSE values for each model

    for name, model in models.items():
        print(f"\nEvaluating model: {name}")
        model.fit(X_train, y_train)  # Fit the model on training data
        metrics = evaluate_model(model, X_val, y_val, dataset_name=name)

        # Store the RMSE value
        rmse_values.append((name, metrics['RMSE']))

        # Select the best model based on the specified metric (RMSE by default)
        if best_metrics is None or metrics[metric] < best_metrics[metric]:
            best_metrics = metrics
            best_model = model

    # Plot RMSE values for all models
    model_names, rmse_vals = zip(*rmse_values)
    plt.figure(figsize=(8, 6))
    plt.bar(model_names, rmse_vals, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Comparison of Model Performance (RMSE)')
    plt.show()

    print(f"\nBest Model: {best_model.__class__.__name__} with {metric}: {best_metrics[metric]:.4f}")
    return best_model, best_metrics

# Forward selection for feature selection
def forward_selection(X_train, y_train, X_val, y_val, max_features=10):
    remaining_features = list(X_train.columns)
    selected_features = []
    best_model = None
    best_metrics = None
    mse_values = []  # To store MSE values during forward selection

    while len(selected_features) < max_features:
        best_score = float('inf')
        best_feature = None

        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train_selected = X_train[current_features]
            X_val_selected = X_val[current_features]

            model = LinearRegression()
            model.fit(X_train_selected, y_train)  # Fit the model here
            y_pred_val = model.predict(X_val_selected)
            metrics =print_regress_metric(y_val, y_pred_val)

            if metrics['MSE'] < best_score:
                best_score = metrics['MSE']
                best_feature = feature
                best_model = model
                best_metrics = metrics

        if best_feature is None:
            break  # Stop if no improvement
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        # Store the MSE after selecting the feature
        mse_values.append(best_score)

    # Plot MSE progression
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mse_values) + 1), mse_values, marker='o', color='b')
    plt.xlabel('Number of Selected Features')
    plt.ylabel('MSE')
    plt.title('Feature Selection Progression')
    plt.grid(True)
    plt.show()

    print("Final selected features:", selected_features)
    return best_model, selected_features, best_metrics


# After performing forward selection, make sure to fit the model before prediction
best_model_forward, selected_features, best_metrics_forward = forward_selection(X_train, y_train, X_val, y_val)

# Now, fit the model on the entire training set with the selected features
X_train_selected = X_train[selected_features]
best_model_forward.fit(X_train_selected, y_train)  # Fit the model with selected features

# Evaluate the model with the selected features on the test set
X_test_selected = X_test[selected_features]
evaluate_model(best_model_forward, X_test_selected, y_test, dataset_name="Test Set after Feature Selection")

# Define models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.01),
    "Ridge Regression": Ridge(alpha=0.1),
    "SVR": SVR(kernel="rbf", C=10, epsilon=0.2)
}


best_model, best_metrics = model_selection(models, X_train, y_train, X_val, y_val)

# Evaluate the best model on the test set
print("\nFinal Test Set Evaluation (Best Model):")
evaluate_model(best_model, X_test, y_test, dataset_name="Test Set")

# Feature selection with forward selection
best_model_forward, selected_features, best_metrics_forward = forward_selection(X_train, y_train, X_val, y_val)

#endproject