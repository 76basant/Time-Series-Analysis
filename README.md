# Time-Series-Analysis


# Comparison Between Linear Regression and Logistic Regression

In machine learning, **Linear Regression** and **Logistic Regression** are two commonly used algorithms. Below is a comparison between the two:

| **Aspect**                     | **Linear Regression**                                         | **Logistic Regression**                                        |
|---------------------------------|---------------------------------------------------------------|----------------------------------------------------------------|
| **Purpose**                     | Predicts a continuous output (real-valued output).            | Predicts a binary or categorical outcome (classification).      |
| **Output Range**                | Outputs can range from negative infinity to positive infinity. | Outputs probabilities between 0 and 1 (for binary classification). |
| **Equation**                    | \( y = \theta_0 + \theta_1 x \)                               | \( p(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x)}} \) (sigmoid function). |
| **Decision Boundary**           | Not explicitly defined.                                       | Decision boundary is where the predicted probability is 0.5.    |
| **Cost Function**               | Mean Squared Error (MSE):                                     | Log Loss (Binary Cross-Entropy):                                |
|                                 | \( \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 \)  | \( -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})] \) |
| **Target Variable**             | Continuous value (e.g., height, weight, temperature).          | Binary (0 or 1, yes or no) or multiclass in the case of extensions. |
| **Assumption about Data**       | Assumes a linear relationship between input variables and output. | Assumes a linear relationship between input variables and the log-odds of the outcome. |
| **Model Type**                  | Regression model.                                              | Classification model.                                           |
| **Application Example**         | Predicting housing prices based on square footage.             | Predicting whether a tumor is malignant or benign based on size. |
| **Gradient Function**           | The gradient is linear: \( \theta = \theta - \alpha \nabla J(\theta) \). | The gradient is non-linear due to the sigmoid function: \( \theta = \theta - \alpha \nabla J(\theta) \). |
| **Interpretation of Coefficients** | Each coefficient represents the change in the output for a one-unit change in the predictor. | Coefficients represent the log-odds of the outcome for a one-unit change in the predictor. |
| **Examples of Use**             | Predicting stock prices, temperature, etc.                    | Email spam detection, disease prediction (tumor classification), etc. |

## Key Differences:

1. **Purpose**:
   - **Linear Regression** predicts continuous values, such as price, temperature, or height.
   - **Logistic Regression** is used for binary classification, such as determining if an email is spam or not.

2. **Equation**:
   - **Linear Regression** models a straight line: \( y = \theta_0 + \theta_1 x \).
   - **Logistic Regression** applies the **sigmoid function**: \( p(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x)}} \), ensuring the output is a probability between 0 and 1.

3. **Cost Function**:
   - **Linear Regression** uses **Mean Squared Error (MSE)** to minimize the difference between actual and predicted values.
   - **Logistic Regression** uses **Log Loss** (cross-entropy loss) to penalize wrong predictions more heavily when they are confident but incorrect.

4. **Decision Boundary**:
   - In **Logistic Regression**, the decision boundary is where the model’s prediction crosses 0.5 (usually used to distinguish classes).
   - In **Linear Regression**, there is no decision boundary concept, as the model is designed to predict continuous values, not classifications.

## Example of Decision Boundary in Logistic Regression:

In **Logistic Regression**, you can find a **decision boundary** where the predicted probability is 0.5. Below is an example of using logistic regression to classify tumors based on their size, and plotting the decision boundary.

### Example Code:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Generate synthetic data for tumor sizes
X = np.linspace(0, 10, 100).reshape(-1, 1)  # Tumor sizes (0 to 10 cm)
y = (X.ravel() > 5).astype(int)  # 1 if tumor size > 5 cm, else 0

# Train logistic regression
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
probabilities = model.predict_proba(X)[:, 1]  # Probability of being malignant

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Tumor Data', edgecolors='k')
plt.plot(X, probabilities, color='red', label='Logistic Regression (Sigmoid Curve)')
decision_boundary = X[np.argmin(np.abs(probabilities - 0.5))][0]
plt.axvline(x=decision_boundary, color='green', linestyle='--', label=f'Decision Boundary at {decision_boundary:.2f} cm')
plt.xlabel('Tumor Size (cm)')
plt.ylabel('Probability of Malignancy')
plt.title('Logistic Regression: Tumor Size vs. Malignancy')
plt.legend()
plt.grid(True)
plt.show()
