# src/predict.py
import joblib
from preprocess import clean_text
import shap
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt


# Load trained pipeline
model = joblib.load("models/fake_news_pipeline.pkl")
class_names = ["REAL", "FAKE"]


def predict_news(text: str):
    """
    Predicts if news is fake or real.
    Returns (label, probability)
    """
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0].max() if hasattr(model, "predict_proba") else None
    label = "FAKE" if pred == 1 else "REAL"
    return label, prob

# ---------- Explainability ----------
def explain_with_lime(text: str, num_features=10):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        clean_text(text), 
        model.predict_proba, 
        num_features=num_features
    )
    exp.show_in_notebook(text=True)
    exp.save_to_file("lime_explanation.html")
    print("✅ LIME explanation saved -> lime_explanation.html")

def explain_with_shap(texts):
    # Get TF-IDF vectorizer from pipeline
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    # Build SHAP explainer
    explainer = shap.Explainer(clf, vectorizer.transform)
    shap_values = explainer(texts)

    # Save summary plot
    shap.summary_plot(shap_values, show=False)
    plt.savefig("shap_summary.png")
    print("✅ SHAP summary saved -> shap_summary.png")


if __name__ == "__main__":
    sample = "Aliens have landed in New York last night!"
    label, prob = predict_news(sample)
    print(f"News: {sample}\nPrediction: {label}, Confidence: {prob:.2f}")

    # Run explainability
    explain_with_lime(sample)
    explain_with_shap([sample])
