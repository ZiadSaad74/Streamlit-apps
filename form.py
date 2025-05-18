import streamlit as st
import joblib
from model_functions import clean_text, ask_llm, job_descriptions, stop_words
import datetime
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

model = joblib.load("model.pkl")  
vectorizer = joblib.load("vec.pkl")  

programs = ["Dentistry", "HR", "Internship-pharma", "pmo", "Quality", "Software", "Physiotherapy", "opd lead"]

majors = ["Computer Science", "Data Science", "Artificial Intelligence", "Cybersecurity", "Software Engineering",
           "Electronics & Communications Engineering", "Civil Engineering", "Architecture", "Mechatronics", 'Veterinary & Animal Science',
           "Biomedical Engineering", "Nanotechnology", "Medicine", "Dentistry", "Pharmacy", "Nursing", "Physiotherapy",
           "Medical Laboratory Sciences", "Radiology", "Nutrition & Dietetics", "Public Health", "Business Administration",
           "Accounting", "Finance", "Marketing", "Human Resources Management", "Entrepreneurship", "International Business", "Economics", 
           "Supply Chain Management", "E-Commerce", "Psychology", "Political Science", "Anthropology", "Media & Communication Studies", 
           "Graphic Design", "Interior Design", "Photography", "Law", "International Relations", "Public Administration", "Primary Education", "Special Education", "Educational Technology", "Curriculum & Instruction", "Educational Leadership", "Social Work", "Criminology", "Human Development", "Agricultural Sciences", "Food Science & Technology", "Hotel and Tourism Management"]

majors.sort()
majors.append("Other")

st.title("Internships programs Recommendation System")

with st.form("course_form"):
    name = st.text_input("Name", placeholder="Enter your full name", max_chars=150)
    age = st.date_input("Date of birth", min_value=datetime.date(1980, 1, 1), max_value=datetime.date.today())
    education_status = st.selectbox("Education status", options=["Graduated", "Student"])
    major = st.selectbox("Select your major", majors)
    job_title = st.text_input("Job Title", placeholder="Enter your job title", max_chars=100)
    selected_course = st.selectbox("Select the internship program", programs)
    submit = st.form_submit_button("Submit")

if submit:
    if not all([name, age, education_status, major, job_title, selected_course]):
        st.error("All fields are required.")
    else:
        original_job_title = job_title
        job_title = clean_text(str(job_title))
        job_title_vector = vectorizer.transform([job_title])
        prediction = model.predict(job_title_vector)[0]
        registration_date = datetime.datetime.now().date()

        if prediction.lower() != selected_course.lower():
            if prediction == "Not Match":
                st.success("❌ Not matched")
            else:
                st.success(f"❌ Not matched, but matched with **{prediction}**")
        else:
            st.success("✅ Matched")

        result = ask_llm(original_job_title, job_descriptions)
        st.success(f"Matched Course (LLM): {result}")

        creds = Credentials.from_service_account_file(r"C:\Users\ziad.saad\Desktop\cities\focal-welder-386600-55fd36951c74.json")
        client = gspread.authorize(creds)
        sheet = client.open("interns").sheet1

        # Insert header if sheet is empty
        if len(sheet.get_all_values()) == 0:
            sheet.insert_row(["Registration Date", "Name", "DOB", "Education Status", "Major", "Job Title", "Selected Course", "Prediction", "LLM Result"], 1)

        sheet.append_row([
            str(registration_date),
            name,
            str(age),
            education_status,
            major,
            original_job_title,
            selected_course,
            prediction,
            result
        ])

        st.success("✅ Data submitted successfully to Google Sheets!")
