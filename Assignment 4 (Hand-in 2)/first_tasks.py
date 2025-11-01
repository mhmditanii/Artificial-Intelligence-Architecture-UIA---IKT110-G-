import numpy as np
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

pio.renderers.default = "browser"

x_train = np.arange(0.0, 1.0, 0.025)
y_train = 0.4 + x_train * 0.55 + np.random.randn(x_train.shape[0]) * 0.2

theta = 0.34


def model(theta, x_train, y_intercept: float = 0):
    return theta * x_train + y_intercept


y_model = model(theta, x_train)

fig = px.scatter(x=x_train, y=y_train, title="Train Data vs Model")
fig.add_scatter(x=x_train, y=y_model, mode="lines", name=f"f_theta(x), theta={theta}")
# fig.show()


def cal_loss(y_true, y_pred):
    m = len(y_true)
    mse = (1 / m) * np.sum((y_true - y_pred) ** 2)
    return mse


results = []
for theta in np.arange(0, 2, 0.01):
    y_model = model(theta, x_train)
    loss = cal_loss(y_train, y_model)
    results.append([theta, loss])

results = np.array(results)

thetas = results[:, 0]
losses = results[:, 1]

# plt.plot(thetas, losses, label="Loss plot")
# plt.xlabel("Theta")
# plt.ylabel("Loss")
# plt.title("Loss Function")
# plt.legend()
# plt.show()

# ----------- Task 2c ---------------

x_train_c = np.linspace(0.5, 1, 30)

rng = np.random.default_rng(0)
y_train_c = 0.95 + 0.20 * x_train_c + rng.normal(0, 0.03, size=30)

# print("X values generated : ")
# print(x_train_c)
# print("--------------------------------")
# print("Y values generated : ")
# print(y_train_c)
# print("--------------------------------")

b = [0.1, 0.75, 1.5]
a_min = -1
a_max = 1.01

"""
for i in range(3):
    loss_res = []
    for theta in np.arange(a_min, a_max, 0.1):
        result = model(theta, x_train_c, b[i])
        loss = cal_loss(y_train_c, result)
        loss_res.append([theta, loss])
    loss_res = np.array(loss_res)

    thetas = loss_res[:, 0]
    losses = loss_res[:, 1]

    plt.figure(figsize=(8, 5))
    plt.plot(thetas, losses, label=f"Loss plot for b = {b[i]}")
    plt.xlabel("Theta")
    plt.ylabel("Loss")
    plt.title(f"Loss Function for b = {b[i]}")
    plt.xlim([-1, 1])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""


# --------- Task 3 -----------
