

from flask import Flask, request, render_template
import joblib
from text_cleaner import TextCleaner   # এখন ইম্পোর্ট হবে এখানে থেকে

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("final_pipeline.pkl")

@app.route("/")
def home():
    result = ""
    model_preds = {'lr': '-', 'knn': '-', 'nb': '-'}
    majority_msg = ""
    return render_template("index.html",
                           result=result,
                           model_preds=model_preds,
                           majority_msg=majority_msg)

@app.route("/predict", methods=["POST"])
def predict():
    result = ""
    model_preds = {}
    majority_msg = ""

    user_text = request.form.get("input_text")
    if user_text:
        cleaner = model.named_steps['cleaner']
        vectorizer = model.named_steps['vectorizer']
        voting_clf = model.named_steps['classifier']

        cleaned = cleaner.transform([user_text])
        features = vectorizer.transform(cleaned)

        pred_labels = []
        for key, clf in voting_clf.named_estimators_.items():
            pred = clf.predict(features)[0]
            label = "Positive" if pred == 1 else "Negative"
            model_preds[key] = label
            pred_labels.append(label)

        majority = max(set(pred_labels), key=pred_labels.count)
        count = pred_labels.count(majority)
        majority_msg = f"{count} out of {len(pred_labels)} models predict '{majority}'"
        result = majority
    else:
        result = "Please enter some text!"
        model_preds = {'lr': '-', 'knn': '-', 'nb': '-'}
        majority_msg = ""

    return render_template("index.html",
                           result=result,
                           model_preds=model_preds,
                           majority_msg=majority_msg)

if __name__ == "__main__":
    app.run(debug=True)
