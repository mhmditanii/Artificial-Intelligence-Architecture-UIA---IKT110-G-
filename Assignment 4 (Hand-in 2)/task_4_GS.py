import numpy as np
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"


def predict(theta, xs):
    return np.dot(xs, theta)


def J_squared_residual(theta, xs, y):
    h = predict(theta, xs)
    sr = ((h - y) ** 2).sum()
    return sr


def gradient_J_squared_residual(theta, xs, y):
    h = predict(theta, xs)
    grad = np.dot(xs.transpose(), (h - y))
    return grad


# the dataset (already augmented so that we get a intercept coef)
# remember: augmented x -> we add a colum of 1's instead of using a bias term.
data_x = np.array([[1.0, 0.5], [1.0, 1.0], [1.0, 2.0]])
data_y = np.array([[1.0], [1.5], [2.5]])
n_features = data_x.shape[1]

# variables we need
theta = np.zeros((n_features, 1))
learning_rate = 0.1
m = data_x.shape[0]

print("data shape 0", data_x.shape[0])
print("data shape 1", data_x.shape[1])

# run GD
j_history = []
n_iters = 10
for it in range(n_iters):
    j = J_squared_residual(theta, data_x, data_y)
    j_history.append(j)

    theta = theta - (
        learning_rate * (1 / m) * gradient_J_squared_residual(theta, data_x, data_y)
    )

# print("theta shape:", theta.shape)

# append the final result.
j = J_squared_residual(theta, data_x, data_y)
j_history.append(j)
# print("The L2 error is: {:.2f}".format(j))


# find the L1 error.
y_pred = predict(theta, data_x)
l1_error = np.abs(y_pred - data_y).sum()
# print("The L1 error is: {:.2f}".format(l1_error))


# Find the R^2
# if the data is normalized: use the normalized data not the original data (task 3 hint).
# https://en.wikipedia.org/wiki/Coefficient_of_determination
u = ((data_y - y_pred) ** 2).sum()
v = ((data_y - data_y.mean()) ** 2).sum()
# print("R^2: {:.2f}".format(1 - (u / v)))


# plot the result
fig = px.line(j_history, title="J(theta) - Loss History")
fig.show()
