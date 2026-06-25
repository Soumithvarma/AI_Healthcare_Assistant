
import joblib
import pandas as pd


# ==========================================
# LOAD SAVED MODEL
# ==========================================

model = joblib.load("backend/models/risk_model.pkl")


# ==========================================
# RISK PREDICTION FUNCTION
# ==========================================

def predict_risk(symptoms, duration):

    # Convert list → string
    symptom_text = " ".join(symptoms)

    # Create dataframe
    sample = pd.DataFrame({
        "symptom_text": [symptom_text],
        "duration_days": [duration]
    })

    # Predict class
    prediction = model.predict(sample)[0]

    # Predict probabilities
    probabilities = model.predict_proba(sample)[0]

    # Store scores
    prob_dict = {}

    for label, prob in zip(model.classes_, probabilities):

        prob_dict[label] = round(prob * 100, 2)

    # ======================================
    # SMART RISK INTERPRETATION
    # ======================================

    sorted_probs = sorted(
        prob_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_label, top1_score = sorted_probs[0]
    top2_label, top2_score = sorted_probs[1]

    difference = top1_score - top2_score

    if difference < 10:

        final_risk = (
            f"{top2_label.capitalize()} "
            f"to "
            f"{top1_label.capitalize()} Risk"
        )

    else:

        final_risk = f"{top1_label.capitalize()} Risk"

    return {
        "prediction": final_risk,
        "probabilities": prob_dict
    }

