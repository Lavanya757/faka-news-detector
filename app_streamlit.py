# app_streamlit.py
import streamlit as st
import joblib
from src.preprocess import clean_text

model = joblib.load("models/fake_news_pipeline.pkl")
st.title("Fake News Detector — Demo")
text = st.text_area("Paste a news item here", height=200)
if st.button("Check"):
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0].max() if hasattr(model, "predict_proba") else None
    label = "FAKE" if pred==1 else "REAL"
    st.markdown(f"**Prediction:** {label}")
    if prob is not None:
        st.markdown(f"**Confidence:** {prob:.2f}")



        # Show explanations
        st.subheader("Explainability")

        expander = st.expander("🔎 LIME Explanation (local)")
        expander.write("LIME highlights words contributing to FAKE vs REAL classification.")
        if st.button("Generate LIME"):
            from lime.lime_text import LimeTextExplainer
            explainer = LimeTextExplainer(class_names=["REAL","FAKE"])
            exp = explainer.explain_instance(
                cleaned,
                model.predict_proba,
                num_features=10
            )
            exp.save_to_file("lime_explanation.html")
            st.markdown("✅ LIME explanation generated! Open [lime_explanation.html](lime_explanation.html)")

        expander2 = st.expander("📊 SHAP Explanation (global & local)")
        expander2.write("SHAP explains feature importance across predictions.")
        if st.button("Generate SHAP"):
            import shap, matplotlib.pyplot as plt
            explainer = shap.Explainer(model.named_steps["clf"], model.named_steps["tfidf"].transform)
            shap_values = explainer([cleaned])
            shap.plots.text(shap_values[0])
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(bbox_inches='tight')
