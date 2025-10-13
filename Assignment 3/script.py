dep_mu = df['dep_min'].mean()
dep_sigma = df['dep_min'].std()
arr_mean = df['arr_min'].mean()
arr_std = df['arr_min'].std()

def scale_input(dep_min):
    return (dep_min - dep_mu) / dep_sigma

def unscale_arrival_time(scaled_time):
    return (scaled_time * arr_std) + arr_mean

def module_linear(xs_input, theta):
    X = np.hstack([np.ones((xs_input.shape[0], 1)), xs_input])
    return X @ theta

def module_polynomial(xs_input, theta):
    X = np.hstack([np.ones((xs_input.shape[0], 1)), xs_input, xs_input**2])
    return X @ theta

def module_tangent(xs_input, theta):
    X = np.hstack([np.ones((xs_input.shape[0], 1)), np.tan(xs_input)])
    return X @ theta

def module_polytan(xs_input, theta):
    X = np.hstack([np.ones((xs_input.shape[0], 1)), xs_input, xs_input**2, np.tan(xs_input)])
    return X @ theta


def regression_loss_function(y_hat, y_true):
    return np.sum((y_hat - y_true)**2)

def sample_theta(size_of_theta, min, max):
    theta = np.random.uniform(min, max, size=size_of_theta)
    return theta

def fortuna_algorithm(xs_data, y_true_data, module, size_of_theta):
    best_loss, best_theta = float('inf'), None
    for i in range(100000):
        curr_theta = sample_theta(size_of_theta, -1.0, 1.0)
        y_hat = module(xs_data, curr_theta)
        curr_loss = regression_loss_function(y_hat, y_true_data)
        if curr_loss < best_loss:
            best_loss, best_theta = curr_loss, curr_theta
    return best_theta, best_loss

N_OUTPUTS = 1 
models_to_train = {
    "Linear":     {"module": module_linear, "theta_shape": (2, N_OUTPUTS)},
    "Polynomial": {"module": module_polynomial, "theta_shape": (3, N_OUTPUTS)},
    "Tangent":    {"module": module_tangent, "theta_shape": (2, N_OUTPUTS)},
    "PolyTan":    {"module": module_polytan, "theta_shape": (4, N_OUTPUTS)}
}

route_names = df['road'].unique()
best_models_per_route = {}

print("Starting route-by-route model training...\n")

for route in route_names:
    print(f"--- Finding Best Model for Route: [{route}] ---")
    
    route_df = df[df['road'] == route]
    xs_route = route_df[['dep_scaled']].values
    y_true_route = route_df[['arr_scaled']].values
    
    route_best_loss = float('inf')
    route_best_model_name = ""
    route_best_theta = None

    for name, model_info in models_to_train.items():
        best_theta, final_loss = fortuna_algorithm(
            xs_route, y_true_route, model_info["module"], model_info["theta_shape"]
        )
        print(f"  - Testing {name}: loss = {final_loss:.4f} , theta = {best_theta}")
        if final_loss < route_best_loss:
            route_best_loss = final_loss
            route_best_model_name = name
            route_best_theta = best_theta
            
    best_models_per_route[route] = {
        "model_name": route_best_model_name,
        "theta": route_best_theta,
        "module": models_to_train[route_best_model_name]["module"],
        "loss": route_best_loss
    }

    # use RMSE in case I have used the mean loss function (and not sum of squared errors)
    #rmse_scaled = np.sqrt(route_best_loss)
    #rmse_minutes = rmse_scaled * arr_std
    print(f" Best model for [{route}] is '{route_best_model_name}' "
          f"with loss {route_best_loss:.4f} (scaled), "
          #f"RMSE = {rmse_minutes:.2f} minutes, "
          f"theta = {route_best_theta}\n")


def predict_best_route_and_time(departure_time_scaled):
    route_predictions = []
    
    for route_name, model_info in best_models_per_route.items():
        module_func = model_info["module"]
        theta = model_info["theta"]
        input_vector = np.array([[departure_time_scaled]])
        predicted_scaled_time = module_func(input_vector, theta)
        predicted_minutes = unscale_arrival_time(predicted_scaled_time[0][0])
        route_predictions.append({"route": route_name, "time": predicted_minutes})

    best_prediction = min(route_predictions, key=lambda x: x["time"])  
    return best_prediction["route"], int(best_prediction["time"])


#departure_input_minutes = 600 # 10:00 AM
#departure_input_scaled = scale_input(departure_input_minutes)
#
#predicted_route, predicted_arrival = predict_best_route_and_time(departure_input_scaled)
#
#print("--- Example Prediction ---")
#print(f"For a raw departure time of: {departure_input_minutes} minutes ({departure_input_minutes // 60:02d}:{departure_input_minutes % 60:02d})")
#print(f" -> Scaled value is: {departure_input_scaled:.4f}")
#print(f"Predicted Best Route: {predicted_route}")
#print(f"Predicted Arrival Time: {predicted_arrival // 60:02d}:{predicted_arrival % 60:02d}")

