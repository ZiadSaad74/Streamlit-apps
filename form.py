import streamlit as st
import joblib
from model_functions import clean_text, ask_llm, job_descriptions, stop_words, joblib
import datetime
import pandas as pd
import os

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
    name = st.text_input("Name", placeholder= "Enter your full name", max_chars=150)
    age = st.date_input("Date of birth", help= "Enter your date of birth", min_value= datetime.date(1980,1,1), max_value= datetime.date.today())
    education_status = st.selectbox("Education status", options= ['Graduated', "Student"])
    major = st.selectbox("Select your major", majors)
    job_title = st.text_input("Job Title", placeholder="Enter your job title", max_chars=100)
    selected_course = st.selectbox("Select the internship program", programs)
    submit = st.form_submit_button("Submit")

if submit:
    result = 0
    if (not job_title) or (not name) or (not age) or (not education_status) or (not selected_course) or (not job_title) :
        st.error("Fields are required.")
    else:
        original_job_title = job_title
        job_title = clean_text(str(job_title))

        job_title_vector = vectorizer.transform([job_title])

        prediction = model.predict(job_title_vector)[0] 

        if prediction.lower() != selected_course.lower():
            if prediction == "Not Match":
                st.success(f"❌ Not matched")
            else:
                st.success(f"❌ Not matched, but matched with **{prediction}**")
        else:
                st.success(f"✅ Matched")

    result = ask_llm(original_job_title,job_descriptions)
    st.success(f"Matched Course: {result}")

    data_to_save = {
    "Registeration date": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "Name": [name],
    "Age": [datetime.date.today().year - age.year],
    "Education level": [education_status],
    "Job Title": [original_job_title],
    "Major": [major],
    "Selected program": [selected_course],
    "Best program": [prediction],
    "Best course": [result]}

    df_new = pd.DataFrame(data_to_save)

    file_path = "submissions.xlsx"

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df_existing = pd.read_excel(file_path)
        except Exception as e:
            st.warning(f"Warning reading existing Excel: {e}")
            df_existing = pd.DataFrame()
    else:
        df_existing = pd.DataFrame()

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(file_path, index=False)