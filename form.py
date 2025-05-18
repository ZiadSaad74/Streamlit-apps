import streamlit as st
import joblib
from model_functions import clean_text, ask_llm, job_descriptions, stop_words
import datetime
import pandas as pd

# Load the model and vectorizer
model = joblib.load("model.pkl")  
vectorizer = joblib.load("vec.pkl")  

# Dropdown options
programs = ["Dentistry", "HR", "Internship-pharma", "pmo", "Quality", "Software", "Physiotherapy", "opd lead"]
majors = ["Computer Science", "Data Science", "Artificial Intelligence", "Cybersecurity", "Software Engineering",
          "Electronics & Communications Engineering", "Civil Engineering", "Architecture", "Mechatronics", 
          'Veterinary & Animal Science', "Biomedical Engineering", "Nanotechnology", "Medicine", "Dentistry", 
          "Pharmacy", "Nursing", "Physiotherapy", "Medical Laboratory Sciences", "Radiology", 
          "Nutrition & Dietetics", "Public Health", "Business Administration", "Accounting", "Finance", 
          "Marketing", "Human Resources Management", "Entrepreneurship", "International Business", "Economics", 
          "Supply Chain Management", "E-Commerce", "Psychology", "Political Science", "Anthropology", 
          "Media & Communication Studies", "Graphic Design", "Interior Design", "Photography", "Law", 
          "International Relations", "Public Administration", "Primary Education", "Special Education", 
          "Educational Technology", "Curriculum & Instruction", "Educational Leadership", "Social Work", 
          "Criminology", "Human Development", "Agricultural Sciences", "Food Science & Technology", 
          "Hotel and Tourism Management"]
majors.sort()
majors.append("Other")

# Title
st.markdown("## 🎯 Internship Program Recommendation System")
st.markdown("Fill in the details below to get a suitable match for your job title.")

# Form
with st.form("course_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("👤 Name", placeholder="Enter your full name", max_chars=150)
        education_status = st.selectbox("🎓 Education status", options=['Graduated', "Student"])
        major = st.selectbox("📘 Your Major", majors)
    with col2:
        age = st.date_input("🎂 Date of birth", help="Enter your date of birth", min_value=datetime.date(1980, 1, 1), max_value=datetime.date.today())
        job_title = st.text_input("💼 Job Title", placeholder="Enter your job title", max_chars=100)
        selected_course = st.selectbox("🎯 Select your preferred internship program", programs)

    submit = st.form_submit_button("🚀 Submit")

# Submission Logic
if submit:
    if not all([name, age, education_status, major, job_title, selected_course]):
        st.error("⚠️ All fields are required.")
    else:
        original_job_title = job_title
        cleaned_title = clean_text(str(job_title))
        job_title_vector = vectorizer.transform([cleaned_title])
        prediction = model.predict(job_title_vector)[0]

        st.markdown("---")
        st.markdown("### 📋 Result Summary")

        if prediction.lower() != selected_course.lower():
            if prediction.lower() == "not match":
                st.error("❌ No matching internship found for this job title.")
            else:
                st.warning(f"❌ Not matched with selected course.\n\n✅ Best match: **{prediction}**")
        else:
            st.success(f"✅ Great! Your selected program **{selected_course}** matches your job title.")

        # LLM output
        st.markdown("### 🤖 Course Suggestion by AI")
        result = ask_llm(original_job_title, job_descriptions)
        st.info(f"**Suggested Course: ** {result}")
