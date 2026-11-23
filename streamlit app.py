import streamlit as st
import joblib
import re
from pathlib import Path

logo_path = str(Path(__file__).with_name("logo.png.jpg"))
#Set application title
st.set_page_config(
    page_title="SMS Spam Detector", 
    page_icon=logo_path,
    layout="centered")
#Load the trained model and vectorizer
model = joblib.load("sms_spam_detector_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

#crete text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
#Application heading
st.image(logo_path, width=250)
st.title("SMS Spam Detector")
st.write("Enter an SMS message below to determine whether it is Spam or Not Spam."
)

#create message input box
message = st.text_area(
    "Enter your SMS message:",
    placeholder= "Type your message here..."
)
#create prediction buttion
if st.button("Predict Message"):
    if message.strip() == "":
        st.warning("Please enter an SMS message.")
    else:
        #clean the message
        cleaned_message = clean_text(message)
        #predict message category
        message_tfidf = tfidf.transform([cleaned_message])
        probabilities = model.predict_proba(message_tfidf)[0]
        predicted_index = probabilities.argmax()
        prediction = model.classes_[predicted_index]
        confidence = probabilities[predicted_index]
        confidence_percentage = confidence * 100

        #display prediction
        if prediction == 1:
            st.error("SPAM MESSAGE DETECTED")
        else:
            st.success("THIS IS NOT SPAM")
            #Display confidence score
            st.subheader("Prediction Confidence")

            st.progress(float(confidence))

            st.write(f"confidence:{confidence_percentage:.2f}%")
            

            st.caption(
                "Confidence represents the model's estimated"
                "probability, not a guarrantee of correctness."
            )
