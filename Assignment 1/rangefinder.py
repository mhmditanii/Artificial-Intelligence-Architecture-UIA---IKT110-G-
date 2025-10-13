from flask import Flask, render_template_string, request
import math

app = Flask(__name__)

LEARNED_INTERCEPT = 23.787587
LEARNED_COEFS = [2.252505, -1.944333, 3.939864]
TRAINING_MAE = 10.01 

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Dist 'O' Meter Calculator </title>
    <style>
        body { font-family: Arial, sans-serif; background: linear-gradient(135deg,#dfe9f3 0%,#ffffff 100%); 
               display:flex;justify-content:center;align-items:center;height:100vh;margin:0;padding:20px;}
        .container { max-width:520px;width:100%;padding:25px;background:#fff;border-radius:12px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.15);}
        h1 { text-align:center;font-size:22px;margin-bottom:20px;color:#333; }
        label { display:block;margin-top:12px;font-weight:bold;color:#444; }
        input[type=number] { width:100%;padding:8px;margin-top:5px;border-radius:6px;border:1px solid #ccc;box-sizing:border-box; }
        button { margin-top:20px;width:100%;padding:12px;background:#4CAF50;color:white;border:none;border-radius:6px;font-size:16px;cursor:pointer; }
        button:hover { background:#45a049; }
        .result { margin-top:20px;padding:15px;background:#eef6ff;border-left:5px solid #4CAF50;border-radius:6px;font-size:15px; }
        .instructions { margin-top:15px;font-size:13px;color:#666;background-color:#f8f9fa;padding:10px;border-radius:6px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dist 'O' Meter</h1>
        <form method="post">
            <label>Angle from Point B (degrees):</label>
            <input type="number" step="0.1" name="angle_b" required>
            <label>Number of Steps (heel-to-toe):</label>
            <input type="number" step="1" name="steps" required>
            <label>Your Shoe Size (EU):</label>
            <input type="number" step="1" name="shoe_size_eu" required>
            <button type="submit">Calculate Distance</button>
        </form>

        {% if distance is not none %}
        <div class="result">
            <b>Theoretical distance:</b> {{ theoretical }} m <br>
            <b>Corrected model estimate :</b> {{ distance }} m <br>
            <b>Estimated error margin:</b> ± {{ error }} m 
        </div>
        {% endif %}

        <div class="instructions">
            <b>How to use:</b>
            <ol>
                <li>At Point A, face the target and walk perpendicular to that line counting steps to Point B.</li>
                <li>At B, measure the angle between your baseline and the line to the target and enter it here.</li>
                <li>Enter steps (heel-to-toe) and your EU shoe size. The app applies a learned correction and shows an error margin.</li>
            </ol>
        </div>
    </div>
</body>
</html>
"""

def apply_correction(theoretical, angle, footsteps):
    return LEARNED_INTERCEPT + LEARNED_COEFS[0]*theoretical + LEARNED_COEFS[1]*angle + LEARNED_COEFS[2]*footsteps

@app.route("/", methods=["GET", "POST"])
def index():
    distance = None
    error = None
    theoretical = None

    if request.method == "POST":
        try:
            angle_b_deg = float(request.form["angle_b"])
            steps = int(request.form["steps"])
            shoe_size_eu = float(request.form["shoe_size_eu"])
            shoe_size_cm = shoe_size_eu * (2.0/3.0)
            shoe_size_m = shoe_size_cm / 100.0
            baseline = steps * shoe_size_m
            angle_b_rad = math.radians(angle_b_deg)
            if abs(math.tan(angle_b_rad)) < 1e-9:
                distance = None
                error = "Angle too small or zero; measurement invalid."
            else:
                theoretical_val = baseline / math.tan(angle_b_rad)
                predicted = apply_correction(theoretical_val, angle_b_deg, steps)
                theoretical = round(theoretical_val, 2)
                distance = round(predicted, 2)
                error = round(TRAINING_MAE, 2)

        except (ValueError, ZeroDivisionError):
            distance = None
            error = "Error in input. Please check your numbers."

    return render_template_string(HTML, distance=distance, theoretical=theoretical, error=error)

if __name__ == "__main__":
    app.run(debug=True)

