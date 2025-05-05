import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

model = joblib.load("model.pkl")  
vectorizer = joblib.load("vec.pkl")  

courses = ["Dentistry", "HR", "Internship-pharma", "pmo", "Quality", "Software", "Physiotherapy", "opd lead"]

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = text.strip()
    text = text.lower()  
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    final = " ".join(tokens)
    return final

st.title("Internships programs Recommendation System")

with st.form("course_form"):
    job_title = st.text_input("Job Title", placeholder="Enter your job title", max_chars=100)
    selected_course = st.selectbox("Select a Course", courses)
    submit = st.form_submit_button("Submit")

if submit:
    if not job_title:
        st.error("Job title is required.")
    else:
        job_title = clean_text(str(job_title))

        job_title_vector = vectorizer.transform([job_title])

        prediction = model.predict(job_title_vector)[0]  # make sure to pass the vectorized input

        if job_title.lower() in ['frontend', 'front end', 'backend','back end', 'full stack', 'fullstack'] and selected_course == 'Software':
            st.success(f"✅ Matched")

        elif job_title.lower() in ['frontend', 'front end', 'backend','back end', 'full stack', 'fullstack'] and selected_course != 'Software':
            st.success(f"❌ Not matched, but matched with **Software**")
        else:
            if prediction.lower() != selected_course.lower():
                if prediction == "Not Match":
                    st.success(f"❌ Not matched")
                else:
                    st.success(f"❌ Not matched, but matched with **{prediction}**")
            
            else:
                st.success(f"✅ Matched")
