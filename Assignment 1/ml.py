import math
import random


footsteps = [11, 13, 15, 14, 23, 22, 10, 9, 7, 9]
angles = [10, 7.5, 20, 5, 10, 5, 10, 5, 12, 15]
shoe_eu = [43] * 10
real_dist = [90, 100, 70, 170, 180, 270, 110, 110, 60, 35]


shoe_len_m = [s * (2/3) / 100 for s in shoe_eu]


baselines = [f * l for f, l in zip(footsteps, shoe_len_m)]

theoretical = []
for b, a in zip(baselines, angles):
    rad = math.radians(a)
    if abs(math.tan(rad)) < 1e-9:
        theoretical.append(0.0)
    else:
        theoretical.append(b / math.tan(rad))


def my_model(theta, x1, x2, x3):
    return theta[0] + theta[1]*x1 + theta[2]*x2 + theta[3]*x3


def my_loss(y_hat, y):
    return (y_hat - y) ** 2

best_loss = float("inf")
best_theta = None


for guess in range(50000): 
    theta0 = random.uniform(-50, 50)  
    theta1 = random.uniform(-5, 5)    
    theta2 = random.uniform(-5, 5)    
    theta3 = random.uniform(-5, 5)    

    total_loss = 0.0
    for x1, x2, x3, y in zip(theoretical, angles, footsteps, real_dist):
        y_hat = my_model([theta0, theta1, theta2, theta3], x1, x2, x3)
        total_loss += my_loss(y_hat, y)

    if total_loss < best_loss:
        best_loss = total_loss
        best_theta = [theta0, theta1, theta2, theta3]
        print("new best loss:", best_loss, "theta:", best_theta)

print("\n final results")
print("Best loss:", best_loss)
print("Best theta:", best_theta)

print("\npredictions vs real:")
for x1, x2, x3, y in zip(theoretical, angles, footsteps, real_dist):
    y_hat = my_model(best_theta, x1, x2, x3)
    print(f"Pred: {y_hat:.1f}, Real: {y}")

