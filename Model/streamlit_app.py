import streamlit as st
import joblib


@st.dialog("Note:")
def modal_dialog():
    st.write("This model can make mistakes so please make sure to not trust it fully")

modal_dialog()

model = joblib.load('Model/scam_model.pkl')
vectorizer = joblib.load('Model/vectorizer.pkl')

st.title("📨 Scam Message Detector :)")
st.markdown("Enter a message to check if it's a scam:")

user_input = st.text_area("")

if st.button("Check"):
    vectorized = vectorizer.transform([user_input])

    probas = model.predict_proba(vectorized)[0]
    ham_prob, spam_prob = probas[0], probas[1]

    confidence = max(ham_prob, spam_prob)
    prediction = model.predict(vectorized)[0]
    if confidence < 0.8:
        prediction = 1 if prediction == 0 else 0

    if prediction == 1:
        st.error("⚠️ This message is a **SCAM**.")
    else:
        st.success("✅ This message is **SAFE**.")

st.markdown("#")
st.markdown("#")
st.markdown("#")

st.write("© 2025 Sreehan Adigopula")

