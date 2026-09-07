import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    pipe = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def predict():

    prediction = None
    bmi = None

    if request.method == "POST":

        age = int(request.form["age"])
        sex = request.form["sex"]

        # Weight & Height
        weight = float(request.form["weight"])
        height = float(request.form["height"])

        # Height (cm) -> Meter
        height = height / 100

        # BMI Calculation
        bmi = weight / (height ** 2)

        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]


        data = np.array([[age, sex, bmi, children, smoker, region]])

        prediction = pipe.predict(data)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2) if prediction is not None else None,
        bmi=round(bmi, 2) if bmi is not None else None
    )


if __name__ == "__main__":
    app.run(debug=True)