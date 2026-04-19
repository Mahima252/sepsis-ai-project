import streamlit as st
import joblib
import numpy as np
import pandas as pd
import uuid
import plotly.graph_objects as go
import plotly.express as px
import json

st.set_page_config(page_title="Sepsis Risk Prediction System", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#e6f2ff,#f5f9ff);
font-family:'Segoe UI';
}

.stButton>button{
background:linear-gradient(90deg,#1e90ff,#0066cc);
color:white;
font-weight:bold;
padding:10px 22px;
border-radius:8px;
border:none;
}

/* Chat UI */
.chat-panel{
width:100%;
background:white;
border-radius:15px;
box-shadow:0 10px 25px rgba(0,0,0,0.2);
padding:10px;
}

.chat-messages{
height:300px;
overflow-y:auto;
padding:10px;
}

.user{text-align:right;margin:6px;}
.bot{text-align:left;margin:6px;}

.userbubble{
background:#c8e6c9;
padding:8px 12px;
border-radius:12px;
display:inline-block;
}

.botbubble{
background:#eeeeee;
padding:8px 12px;
border-radius:12px;
display:inline-block;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("sepsis_model.pkl")

# ---------------- TITLE ----------------
st.title("🩺 Sepsis Risk Prediction System")

st.write("""
Sepsis is a life-threatening condition caused by the body's extreme response to infection.
Early detection through monitoring patient vital signs can prevent organ damage
and improve survival rates.
""")

# ---------------- PATIENT ID ----------------
if "patient_id" not in st.session_state:
    st.session_state.patient_id = str(uuid.uuid4())[:8]

st.subheader(f"Patient ID: {st.session_state.patient_id}")

st.divider()

# ---------------- INPUTS ----------------
st.subheader("Enter Patient Vitals")

hr = st.number_input("Heart Rate", value=80)
temp = st.number_input("Temperature", value=37.0)
resp = st.number_input("Respiratory Rate", value=18)
o2 = st.number_input("Oxygen Saturation", value=96)
wbc = st.number_input("WBC Count", value=7000)

predict = st.button("Predict Sepsis Risk")

# ---------------- DASHBOARD LAYOUT ----------------
col1, col2, col3 = st.columns([1.2,1.5,1])

risk = 0

with col1:

    st.subheader("Patient Vital Monitor")

    st.metric("❤️ Heart Rate",hr)
    st.metric("🌡 Temperature",temp)
    st.metric("🫁 Resp Rate",resp)
    st.metric("🩸 Oxygen",o2)
    st.metric("🧬 WBC",wbc)

with col2:

    st.subheader("Sepsis Risk Prediction")

    if predict:

        data = np.array([[hr,temp,resp,o2,wbc]])

        prediction = model.predict(data)
        probability = model.predict_proba(data)

        risk = probability[0][1] * 100

        st.subheader(f"Risk: {risk:.2f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text':"ICU Risk Meter"},
            gauge={
                'axis':{'range':[0,100]},
                'steps':[
                    {'range':[0,40],'color':'green'},
                    {'range':[40,70],'color':'yellow'},
                    {'range':[70,100],'color':'red'}
                ]
            }
        ))

        st.plotly_chart(fig,use_container_width=True)

        if risk > 70:
            st.markdown("""
            <div style="
            background:red;
            color:white;
            padding:15px;
            text-align:center;
            font-weight:bold;
            border-radius:10px;
            animation: blink 1s infinite;">
            🚨 ICU EMERGENCY ALERT 🚨
            </div>

            <style>
            @keyframes blink {50%{opacity:0;}}
            </style>
            """,unsafe_allow_html=True)

        elif risk > 40:
            st.warning("⚠ Moderate Risk")

        else:
            st.success("✅ Patient Stable")

        # ECG Monitor
        st.markdown("### ECG Monitor")

        t = np.linspace(0,10,500)
        signal = np.sin(5*t) + 0.5*np.sin(15*t)

        ecg = pd.DataFrame({"ECG":signal})

        st.line_chart(ecg)

        # AI Risk Chart
        st.markdown("### AI Risk Factors")

        features = {
            "Heart Rate":hr,
            "Temperature":temp,
            "Resp Rate":resp,
            "Oxygen":o2,
            "WBC":wbc
        }

        fig2 = px.bar(
            x=list(features.keys()),
            y=list(features.values()),
            labels={"x":"Vitals","y":"Values"}
        )

        st.plotly_chart(fig2)

        # History
        if "history" not in st.session_state:
            st.session_state.history=[]

        st.session_state.history.append({
            "HR":hr,
            "Temp":temp,
            "Resp":resp,
            "O2":o2,
            "WBC":wbc,
            "Risk":risk
        })

        report={
            "Patient ID":st.session_state.patient_id,
            "Risk":risk
        }

        st.download_button(
            "Download Patient Report",
            json.dumps(report,indent=4),
            file_name="patient_report.json"
        )

with col3:

    st.subheader("AI Doctor Assistant")

    def doctor_ai(q):

        q=q.lower()

        if "cause" in q:
            return "Sepsis is caused by infections entering the bloodstream."

        elif "symptom" in q:
            return "Symptoms include fever, breathing difficulty and confusion."

        elif "treat" in q:
            return "Treatment includes antibiotics, IV fluids and oxygen therapy."

        elif "what is sepsis" in q:
            return "Sepsis is a life-threatening response to infection."

        else:
            return "Please consult a healthcare professional."

    if "messages" not in st.session_state:
        st.session_state.messages=[
            {"role":"bot","content":"Hello! Ask me about sepsis causes, symptoms or treatment."}
        ]

    chat_html="<div class='chat-panel'><div class='chat-messages'>"

    for msg in st.session_state.messages:

        if msg["role"]=="user":
            chat_html+=f"<div class='user'><span class='userbubble'>👤 {msg['content']}</span></div>"
        else:
            chat_html+=f"<div class='bot'><span class='botbubble'>🤖 {msg['content']}</span></div>"

    chat_html+="</div></div>"

    st.markdown(chat_html,unsafe_allow_html=True)

    with st.form("chat_form",clear_on_submit=True):

        user_input = st.text_input("",placeholder="Type message...")

        send = st.form_submit_button("Send")

        if send and user_input:

            st.session_state.messages.append({"role":"user","content":user_input})

            reply = doctor_ai(user_input)

            st.session_state.messages.append({"role":"bot","content":reply})

            st.rerun()

st.divider()

# ---------------- HISTORY ----------------
st.subheader("Patient Risk Trend")

if "history" in st.session_state:

    df = pd.DataFrame(st.session_state.history)

    st.line_chart(df["Risk"])

st.divider()

# ---------------- MEDICAL INFO ----------------
st.subheader("Medical Information")

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("### Causes")
    st.write("""
• Infection enters bloodstream  
• Immune system overreacts  
• Organs become damaged
""")
    st.image("causes.png")

with c2:
    st.markdown("### Symptoms")
    st.write("""
• Fever and rapid heart rate  
• Shortness of breath  
• Confusion and weakness
""")
    st.image("symptoms.png")

with c3:
    st.markdown("### Treatment")
    st.write("""
• Antibiotics kill infection  
• IV fluids stabilize blood pressure  
• ICU monitoring for severe cases
""")
    st.image("treatment.png")





