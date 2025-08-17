import streamlit as st
import joblib

# ---------- Sidebar State ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"  # Default page

# ---------- Page Navigation Functions ----------
def go_home():
    st.session_state.page = "Home"

def go_about():
    st.session_state.page = "About"

def go_contact():
    st.session_state.page = "Contact"

# ---------- Small Button Styling ----------
st.markdown("""
    <style>
    .sidebar-btn {
        background-color: #f0f2f6;
        padding: 6px 10px;
        margin: 3px 0;
        border-radius: 6px;
        font-size: 14px;
        text-align: left;
        border: none;
        cursor: pointer;
        width: 100%;
    }
    .sidebar-btn:hover {
        background-color: #e0e2e6;
    }
    .active-btn {
        background-color: #4cafef !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📋 Menu")
    st.button("🏠 Home", key="home", on_click=go_home)
    st.button("ℹ️ About", key="about", on_click=go_about)
    st.button("📞 Contact", key="contact", on_click=go_contact)



# ---------- Load Model ----------
model = joblib.load('Model/scam_model.pkl')
vectorizer = joblib.load('Model/vectorizer.pkl')

# ---------- Page Content ----------
if st.session_state.page == "Home":
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

elif st.session_state.page == "About":
    st.title("ℹ️ About")
    st.write("This app uses a trained machine learning model to detect scam messages.")
    st.write("This model can make mistakes so **please make sure to not trust it fully**")
    st.write("I'm still working on improving the model.")
    st.write("Developed by **Sreehan Adigopula**, 2025.")

elif st.session_state.page == "Contact":
    st.title("📞 Contact")
    st.write("For inquiries, reach out via:")
    st.markdown("- Email: `youremail@example.com`")
    st.markdown("- GitHub: [Your GitHub](https://github.com/username)")

# ---------- Footer ----------
st.markdown("#")
st.markdown("#")
st.markdown("#")
st.write("© 2025 Sreehan Adigopula")
