import streamlit as st
import joblib
import datetime
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from model_functions import clean_text, ask_llm, job_descriptions

model = joblib.load("model.pkl")
vectorizer = joblib.load("vec.pkl")

programs = ["Dentistry", "HR", "Internship-pharma", "PMO", "Quality", "Software", "Physiotherapy", "OPD Lead"]
majors = sorted([
    "Computer Science", "Data Science", "Artificial Intelligence", "Cybersecurity", "Software Engineering",
    "Electronics & Communications Engineering", "Civil Engineering", "Architecture", "Mechatronics",
    "Veterinary & Animal Science", "Biomedical Engineering", "Nanotechnology", "Medicine", "Dentistry",
    "Pharmacy", "Nursing", "Physiotherapy", "Medical Laboratory Sciences", "Radiology",
    "Nutrition & Dietetics", "Public Health", "Business Administration", "Accounting", "Finance",
    "Marketing", "Human Resources Management", "Entrepreneurship", "International Business", "Economics",
    "Supply Chain Management", "E-Commerce", "Psychology", "Political Science", "Anthropology",
    "Media & Communication Studies", "Graphic Design", "Interior Design", "Photography", "Law",
    "International Relations", "Public Administration", "Primary Education", "Special Education",
    "Educational Technology", "Curriculum & Instruction", "Educational Leadership", "Social Work",
    "Criminology", "Human Development", "Agricultural Sciences", "Food Science & Technology",
    "Hotel and Tourism Management"
])
majors.append("Other")

st.set_page_config(page_title="Internship Recommendation", layout="centered")
st.title("🎓 Internship Programs Recommendation System")
st.markdown("Please fill out the form to check if your job title matches the selected internship program.")

with st.form("intern_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("👤 Full Name", max_chars=150)
        dob = st.date_input("📅 Date of Birth", min_value=datetime.date(1980,1,1), max_value=datetime.date.today())
        education_status = st.selectbox("🎓 Education Status", ['Graduated', "Student"])
    with col2:
        major = st.selectbox("📚 Major", majors)
        job_title = st.text_input("💼 Job Title", max_chars=100)
        selected_course = st.selectbox("📌 Select Internship Program", programs)

    submit = st.form_submit_button("Submit")

if submit:
    if not all([name, dob, education_status, major, job_title, selected_course]):
        st.error("❗ Please fill in all required fields.")
    else:
        cleaned_title = clean_text(job_title)
        job_title_vector = vectorizer.transform([cleaned_title])
        prediction = model.predict(job_title_vector)[0]
        llm_result = ask_llm(job_title, job_descriptions)

        st.markdown("---")
        st.subheader("🧠 Prediction Result")

        if prediction.lower() == selected_course.lower():
            st.success(f"✅ Matched! Your job title fits the selected program: **{selected_course}**.")
        elif prediction.lower() == "not match":
            st.warning(f"❌ No match found for your job title.")
        else:
            st.info(f"🔎 Not matched with selected course. But model suggests: **{prediction}**")

        st.markdown(f"💡 AI courses Suggestion: **{llm_result}**")

        try:
            creds = Credentials.from_service_account_file("C:/Users/ziad.saad/Desktop/cities/focal-welder-386600-55fd36951c74.json") 
            client = gspread.authorize(creds)
            sheet = client.open("interns").sheet1

            sheet.append_row([
                str(datetime.date.today()), name, str(dob), education_status, major,
                job_title, selected_course, prediction, llm_result
            ])

            st.success("✅ Your data has been saved successfully!")
        except Exception as e:
            st.error(f"Error saving to Google Sheet: {e}")
