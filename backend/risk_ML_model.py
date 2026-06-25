
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer




df = pd.read_csv("C:\\Users\\soumi\\OneDrive\\Desktop\\Arthur Morgan\\AI_assistant\\backend\\dataset\\symptom_risk_dataset_with_duration.csv")

print(df.head())
df = df.dropna()





df["symptom_text"] = df["symptom_text"].str.replace(",", " ")



X = df[["symptom_text", "duration_days"]]
y = df["risk_level"]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



preprocessor = ColumnTransformer([
    ("text", TfidfVectorizer(ngram_range=(1,2)), "symptom_text"),
    ("duration", "passthrough", ["duration_days"])
])




model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])


print("\nTraining model...\n")

model.fit(X_train, y_train)

print("Model trained successfully!\n")




y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")



print("Classification Report:\n")

print(classification_report(y_test, y_pred))




with open("C:\\Users\\soumi\\OneDrive\\Desktop\\Arthur Morgan\\AI_assistant\\backend\\models\\risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as risk_model.pkl\n")



while True:

    user_input = input("Enter symptoms: ")

    if user_input.lower() == "quit":
        break

    duration = int(input("Enter duration in days: "))

  
    user_input = user_input.replace(",", " ")


    sample = pd.DataFrame({
        "symptom_text": [user_input],
        "duration_days": [duration]
    })


    prediction = model.predict(sample)[0]

   
    probabilities = model.predict_proba(sample)[0]

    print("\nConfidence Scores:")

    
    prob_dict = {}

    for label, prob in zip(model.classes_, probabilities):

        percent = prob * 100
        prob_dict[label] = percent

        print(f"{label}: {percent:.2f}%")


    sorted_probs = sorted(
        prob_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_label, top1_score = sorted_probs[0]
    top2_label, top2_score = sorted_probs[1]

    difference = top1_score - top2_score

    
    if difference <= 20:
        final_risk = f"{top2_label.capitalize()} to {top1_label.capitalize()} Risk"
    else:
        final_risk = f"{top1_label.capitalize()} Risk"

    print("\nPredicted Risk:", final_risk)

    print("\n" + "=" * 50 + "\n")