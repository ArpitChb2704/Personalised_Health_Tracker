import gradio as gr
import pandas as pd
import pickle
import numpy as np
import pandas as pd

# Load the model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define the prediction function
def predict_diabetes_risk(pregnancies, glucose, blood_pressure, insulin, bmi, age):
    """
    Predict diabetes risk based on user inputs.
    """
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, insulin, bmi, age]])
    prediction = model.predict(input_data)  # Assuming the model has a `predict` method
    probability = model.predict_proba(input_data)[0][1]  # If available, get prediction probability
    risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
    return f"Risk Level: {risk_level} (Probability: {probability:.2%})"

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Diabetes Risk Predictor")
    gr.Markdown("Enter the following details to predict your risk of diabetes.")

    with gr.Row():
        pregnancies = gr.Number(label="Pregnancies", value=0)
        glucose = gr.Number(label="Glucose Level", value=0)
        blood_pressure = gr.Number(label="Blood Pressure Level", value=0)
        insulin = gr.Number(label="Insulin Level", value=0)
        bmi = gr.Number(label="BMI (Body Mass Index)", value=0)
        age = gr.Number(label="Age", value=0)

    predict_button = gr.Button("Predict Risk")
    output = gr.Textbox(label="Prediction Result")

    predict_button.click(
        predict_diabetes_risk,
        inputs=[pregnancies, glucose, blood_pressure, insulin, bmi, age],
        outputs=output,
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()

