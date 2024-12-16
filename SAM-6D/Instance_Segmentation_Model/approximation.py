import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Actual distance data (x)
x = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5, 7.0, 7.5, 8.0, 8.5, 8.5, 9.0, 9.5, 10.0])

# New actual data (y_actual)
y_actual = np.array([0.000501863, 
                     0.00063301, 
                     0.000953303, 
                     0.001502404, 
                     0.001970513, 
                     0.002551635, 
                     0.002971537, 
                     0.004205938, 
                     0.004592718, 
                     0.005519382, 
                     0.005827187, 
                     0.006047248, 
                     0.005819472, 
                     0.006287639, 
                     0.005619392, 
                     0.005947706, 
                     0.005651437, 
                     0.009693149, 
                     0.008456224,
                     0.009158184])

# Define the linear model
def linear_model(x, a, b):
    return a * x + b

# Define the exponential model
def exp_model(x, a, b):
    return a * np.exp(b * x)

# Fit the linear model
popt_linear, _ = curve_fit(linear_model, x, y_actual)
a_linear, b_linear = popt_linear
y_fit_linear = linear_model(x, a_linear, b_linear)

# Fit the exponential model
popt_exp, _ = curve_fit(exp_model, x, y_actual)
a_exp, b_exp = popt_exp
y_fit_exp = exp_model(x, a_exp, b_exp)

# Calculate the coefficient of determination (R²)
r2_linear = r2_score(y_actual, y_fit_linear)
r2_exp = r2_score(y_actual, y_fit_exp)

# Display the results
print(f'Linear Model Parameters: a = {a_linear}, b = {b_linear}')
print(f'Exponential Model Parameters: a = {a_exp}, b = {b_exp}')
print(f'R² for Linear Model: {r2_linear}')
print(f'R² for Exponential Model: {r2_exp}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y_actual, 'bo', label='Actual Data')  # Actual data
plt.plot(x, y_fit_linear, 'r-', label=f'Linear Fit: y = {a_linear:.6f} * x + {b_linear:.6f}, R² = {r2_linear:.4f}')
plt.plot(x, y_fit_exp, 'g--', label=f'Exponential Fit: y = {a_exp:.6f} * exp({b_exp:.6f} * x), R² = {r2_exp:.4f}')
plt.xlabel('Actual Distance')
plt.ylabel('Data Value')
plt.legend()
plt.title('Linear and Exponential Fits for Actual Data')
plt.grid(True)
plt.show()