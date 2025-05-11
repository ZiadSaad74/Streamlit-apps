import joblib
import string
import nltk
from nltk.corpus import stopwords
import re
from langchain_fireworks import ChatFireworks
from langchain_core.messages import HumanMessage, SystemMessage
import json

job_descriptions = {
    "AI in Healthcare": "Drives AI for healthcare, builds models, analyzes data, ensures compliance. with Skills Python, ML, healthcare knowledge.",
    "AI in Marketing": "Drives AI for marketing, builds models, analyzes data, ensures compliance. with Skills Python, ML, marketing knowledge.",
    "AI in Insurance": "Drives AI for insurance, builds models, analyzes data, ensures compliance. with Skills Python, ML, insurance knowledge.",
    "AI in Engineering": "Drives AI for engineering, builds models, analyzes data, ensures compliance. with Skills Python, ML, engineering knowledge.",
    "BIM Modeling, Coordination & Automation": "Drives bim modeling, coordination & automation for building design, creates BIM models, produces drawings, collaborates. with Skills Revit, CAD, structural engineering.",
    "Structural BIM, STAAD Pro, RAM Connection": "Drives structural bim, staad pro, ram connection for building design, creates BIM models, produces drawings, collaborates. with Skills Revit, CAD, structural engineering.",
    "Technical & Shop Drawing": "Drives technical & shop drawing for building design, creates BIM models, produces drawings, collaborates. with Skills Revit, CAD, structural engineering.",
    "FIDIC Diploma": "Drives fidic diploma for project compliance, advises on contracts or sustainability, conducts audits. with Skills contract law, energy modeling, sustainability.",
    "Energy Modeling + LEED": "Drives energy modeling + leed for project compliance, advises on contracts or sustainability, conducts audits. with Skills contract law, energy modeling, sustainability.",
    "Construction & Prefabrication": "Drives construction & prefabrication for efficient construction, plans workflows, applies lean methods, ensures quality. with Skills project management, lean principles, construction.",
    "5D/7D Lean Optimization": "Drives 5d/7d lean optimization for efficient construction, plans workflows, applies lean methods, ensures quality. with Skills project management, lean principles, construction.",
    "Business Acumen for Engineers": "Drives business strategies for engineers, analyzes trends, aligns projects, develops plans. with Skills business analysis, strategy, communication.",
    "Marketing": "Drives marketing campaigns, develops strategies, creates content, analyzes performance. with Skills marketing tools, creativity, analytics.",
    "Digital Marketing": "Drives digital marketing campaigns, develops strategies, creates content, analyzes performance. with Skills marketing tools, creativity, analytics.",
    "Creative & Art Direction": "Drives creative & art direction campaigns, develops strategies, creates content, analyzes performance. with Skills marketing tools, creativity, analytics.",
    "CRM": "Drives crm for revenue growth, manages relationships, analyzes metrics, sets pricing. with Skills CRM software, data analysis, negotiation.",
    "Growth Intelligence": "Drives growth intelligence for revenue growth, manages relationships, analyzes metrics, sets pricing. with Skills CRM software, data analysis, negotiation.",
    "Sales": "Drives sales for revenue growth, manages relationships, analyzes metrics, sets pricing. with Skills CRM software, data analysis, negotiation.",
    "Pricing Strategy": "Drives pricing strategy for revenue growth, manages relationships, analyzes metrics, sets pricing. with Skills CRM software, data analysis, negotiation.",
    "Event Planning": "Drives event planning for successful execution, plans logistics, coordinates vendors, manages budgets. with Skills organization, communication, multitasking.",
    "Business Innovation": "Drives business innovation for efficiency, identifies opportunities, implements solutions, trains teams. with Skills change management, strategy, leadership.",
    "Digital Transformation": "Drives digital transformation for efficiency, identifies opportunities, implements solutions, trains teams. with Skills change management, strategy, leadership.",
    "Financial Modeling": "Drives financial modeling for forecasting, creates projections, analyzes investments, presents insights. with Skills Excel, financial analysis, forecasting.",
    "HR & Recruitment": "Drives hr & recruitment for efficiency, recruits talent or manages logistics, ensures compliance. with Skills HR or supply chain tools, organization, problem-solving.",
    "Supply Chain Management": "Drives supply chain management for efficiency, recruits talent or manages logistics, ensures compliance. with Skills HR or supply chain tools, organization, problem-solving.",
    "PMP & PMO": "Drives pmp & pmo for project success, plans projects, implements systems, tracks performance. with Skills project management, ERP systems, leadership.",
    "Internal Business Systems": "Drives internal business systems for project success, plans projects, implements systems, tracks performance. with Skills project management, ERP systems, leadership.",
    "Hospital Management Diploma": "Drives hospital management diploma for healthcare delivery, oversees operations, manages staff, ensures compliance. with Skills healthcare management, leadership, regulations.",
    "OPD Management": "Drives opd management for healthcare delivery, oversees operations, manages staff, ensures compliance. with Skills healthcare management, leadership, regulations.",
    "TQM (Total Quality Management)": "Drives tqm (total quality management) for healthcare quality, monitors standards, optimizes cycles, ensures compliance. with Skills quality assurance, financial analysis, regulations.",
    "RCM (Revenue Cycle Management)": "Drives rcm (revenue cycle management) for healthcare quality, monitors standards, optimizes cycles, ensures compliance. with Skills quality assurance, financial analysis, regulations.",
    "Clinical Governance": "Drives clinical governance for healthcare quality, monitors standards, optimizes cycles, ensures compliance. with Skills quality assurance, financial analysis, regulations.",
    "Medical Coding": "Drives medical coding for billing, assigns codes, reviews records, ensures compliance. with Skills medical terminology, coding systems, accuracy.",
    "BLS (Basic Life Support)": "Drives bls (basic life support) in emergencies, delivers interventions, assesses patients, collaborates. with Skills emergency response, medical knowledge, teamwork.",
    "ACLS (Advanced Cardiac Life Support)": "Drives acls (advanced cardiac life support) in emergencies, delivers interventions, assesses patients, collaborates. with Skills emergency response, medical knowledge, teamwork.",
    "ATLS (Advanced Trauma Life Support)": "Drives atls (advanced trauma life support) in emergencies, delivers interventions, assesses patients, collaborates. with Skills emergency response, medical knowledge, teamwork.",
    "POCUS (Point-of-Care Ultrasound)": "Drives pocus (point-of-care ultrasound) for diagnostics, operates ultrasound, interprets images, assists physicians. with Skills ultrasound tech, imaging, patient care.",
    "PALS (Pediatric Advanced Life Support)": "Drives pals (pediatric advanced life support) in emergencies, delivers interventions, assesses patients, collaborates. with Skills emergency response, medical knowledge, teamwork.",
    "Infection Control": "Drives infection control for safety, monitors risks, trains staff, ensures compliance. with Skills infection control, medical procedures, regulations.",
    "Airway Management": "Drives airway management for safety, monitors risks, trains staff, ensures compliance. with Skills infection control, medical procedures, regulations.",
    "Nursing Programs": "Drives nursing programs for patient care, administers treatments, monitors patients, educates. with Skills nursing skills, patient care, empathy.",
    "Software Testing": "Drives software testing for quality, designs tests, reports bugs, collaborates. with Skills testing tools, analytical skills, teamwork.",
    "Data Analysis": "Drives data analysis for insights, processes data, creates visualizations, interprets trends. with Skills Python, SQL, data visualization.",
    "ITI Preparation Exam": "Drives ITI preparation for IT roles, studies fundamentals, practices exams, develops skills. with Skills IT basics, problem-solving, technical aptitude.",
    "Python": "Drives python for applications, writes code, debugs, integrates APIs. with Skills python, problem-solving, version control.",
    "Power Automate": "Drives power automate for workflows, designs apps, integrates Microsoft tools, optimizes. with Skills power automate, Microsoft 365, UX design.",
    "Power Apps": "Drives power apps for workflows, designs apps, integrates Microsoft tools, optimizes. with Skills power apps, Microsoft 365, UX design.",
    "SharePoint & SPFX": "Drives sharepoint & spfx for solutions, builds sites, creates web parts, ensures security. with Skills SharePoint, SPFX, JavaScript.",
    "Flutter & Native App Development": "Drives flutter & native app development for mobile apps, develops apps, tests functionality, deploys. with Skills Flutter, Dart, UI/UX.",
    "Full Stack Web Development": "Drives full stack web development for web apps, develops front/back-end, manages databases, deploys. with Skills JavaScript, Node.js, databases.",
    "PHP": "Drives php for applications, writes code, debugs, integrates APIs. with Skills php, problem-solving, version control.",
    "Cybersecurity": "Drives cybersecurity for system protection, monitors threats, implements security, responds to incidents. with Skills cybersecurity tools, risk assessment, networking.",
    "CCNA": "Drives ccna for network management, configures devices, troubleshoots, ensures security. with Skills networking, Cisco systems, troubleshooting.",
    "AI / Machine Learning": "Drives ai / machine learning for intelligent systems, builds models, preprocesses data, optimizes. with Skills Python, TensorFlow, data science.",
    "IT & Tech Essentials": "Drives it & tech essentials for IT support, troubleshoots systems, manages networks, assists users. with Skills IT basics, troubleshooting, customer service.",
    "Business Acumen for Developers": "Drives business strategies for developers, analyzes trends, aligns projects, develops plans. with Skills business analysis, strategy, communication."}

def ask_llm(title, courses):

    if not isinstance(title,str) or not title.strip():
        return "Not match"
    
    else:
        schema = {
        "type": "object",
        "properties": {
            "output": {
                "type": "string",
                "description": "The single most suitable course name or 'Not match'"}},
        "required": ["output"],
        "additionalProperties": False}

        model = ChatFireworks(
            model="accounts/fireworks/models/llama4-maverick-instruct-basic",temperature=0.1,
            max_tokens=100, fireworks_api_key="fw_3ZTtMu8yiAV962ddVBRTHBAD",top_p=0.3, top_k=3,)

        prompt = f"""
        You are a recommendation system. Your task is to return the most suitable course from a provided list based on a job title.

        Inputs:
        - Job title: "{title}"
        - Course list: {list(courses.keys())}
        
        Instructions:
        - Choose the single best matching course for the job title.
        - If no course is a good match, respond with: {{"output": "Not match" }}
        - Output format must be strictly: {{"output": "<exact course name or Not match>" }}
        - No extra text, explanation, or formatting

        Examples:
        - Job title: "social media specialist" → {{ "output": "Digital Marketing" }}
        - Job title: "electronic engineer" → {{ "output": "CCNA" }}
        - Job title: "Teacher" → {{ "output": "Not match" }}

        Classify this job title: "{title}"
        """

        json_model = model.bind(response_format={"type": "json_object", "schema": schema})
        chat_history = [SystemMessage(content=prompt), HumanMessage(content=title)]

        try:
            response = json_model.invoke(chat_history)
            content = response.content.strip()

            parsed = json.loads(content)
            course_name = parsed.get("output", "Not match")
            return course_name

        except Exception as e:
            return "Not match"

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = str(text)
    text = text.strip()
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)