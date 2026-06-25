import sys
import os
import html
import time

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st

from backend.symptom_extractor import extract_symptoms
from backend.risk_predictor import predict_risk
from backend.rag_chatbot import get_rag_response




st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🩺",
    layout="wide"
)



if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
   st.markdown("""
    <div class="welcome-box">
        <h1>🩺 AI Healthcare Assistant</h1>
        <p>
            Welcome! Describe your symptoms or ask any
            health-related questions.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');


html,
body,
.main {
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
}
 
           .stApp {

    background-color: #000000 !important;

    background-image:

        radial-gradient(
            circle at top left,
            rgba(16,185,129,0.24) 0%,
            rgba(16,185,129,0.10) 28%,
            rgba(16,185,129,0.04) 50%,
            transparent 72%
        ),

        radial-gradient(
            circle at top right,
            rgba(52,255,208,0.20) 0%,
            rgba(52,255,208,0.08) 26%,
            rgba(52,255,208,0.03) 48%,
            transparent 70%
        ),

        linear-gradient(
            180deg,
            #050505,
            #000000
        ) !important;

    background-attachment: fixed;
}
            
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}




header,
footer,
#MainMenu,
[data-testid="stToolbar"],
[data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
}




.block-container {
    padding-top: 1rem !important;

    padding-bottom: 10px !important;

    max-width: 100% !important;
}




[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {

    background: #000000 !important;

    background-color: #000000 !important;

    border: none !important;

    box-shadow: none !important;
}


            
.welcome-box{

  background:

        linear-gradient(
            145deg,

            rgba(8,15,20,0.96) 0%,

            rgba(10,25,22,0.98) 22%,

            rgba(6,78,59,0.95) 55%,

            rgba(8,15,20,0.96) 100%
        );
    width: 50%;

    margin: 40px auto;

    padding: 40px 40px;

    border-radius: 40px;

    text-align: center;

    color: white;

    position: relative;

    animation:
        fadeIn 1.5s ease-in-out,
        glowPulse 3s infinite ease-in-out;

    box-shadow:
        0 0 20px rgba(16,185,129,0.35),
        0 0 40px rgba(16,185,129,0.25),
        0 0 80px rgba(16,185,129,0.18);
}

.welcome-box h1{
    font-size: 30px;
    margin-bottom: 12px;
    font-weight: 700;
}

.welcome-box p{
    font-size: 18px;
    opacity: 0.96;
    line-height: 1.6;
}

@keyframes fadeIn{
    from{
        opacity: 0;
        transform: translateY(20px);
    }
    to{
        opacity: 1;
        transform: translateY(0px);
    }
}
            @keyframes glowPulse {

    0% {

        box-shadow:
            0 0 20px rgba(16,185,129,0.30),
            0 0 40px rgba(16,185,129,0.20),
            0 0 80px rgba(16,185,129,0.12);
    }

    50% {

        box-shadow:
            0 0 35px rgba(16,185,129,0.50),
            0 0 70px rgba(16,185,129,0.35),
            0 0 120px rgba(16,185,129,0.22);
    }

    100% {

        box-shadow:
            0 0 20px rgba(16,185,129,0.30),
            0 0 40px rgba(16,185,129,0.20),
            0 0 80px rgba(16,185,129,0.12);
    }
}



.message-row {
    display: flex;
    margin: 18px 0;
    animation: fadeIn 0.3s ease;
}

.user-row {
    justify-content: flex-end;
}

.assistant-row {
    justify-content: flex-start;
}




.user-bubble {

    background: linear-gradient(
        135deg,
        #10B981,
        #0EA271
    );

    color: white;

    padding: 14px 18px;

    border-radius: 18px 18px 4px 18px;

    max-width: 70%;

    font-size: 15px;

    line-height: 1.5;

    box-shadow:
        0 4px 18px rgba(16,185,129,0.15);

    word-wrap: break-word;
}




.ai-bubble {

    background:
        linear-gradient(
            135deg,
            rgba(25,25,30,0.95),
            rgba(18,18,22,0.95)
        );

    border: 1px solid rgba(255,255,255,0.06);

    color: #EAEAEA;

    padding: 14px 18px;

    border-radius: 18px 18px 18px 4px;

    max-width: 70%;

    font-size: 15px;

    line-height: 1.6;

    backdrop-filter: blur(10px);

    box-shadow:
        0 4px 24px rgba(0,0,0,0.35);

    word-wrap: break-word;
}




.stChatInput {
    position: fixed !important;

    bottom: 8px !important;

    left: 50% !important;

    transform: translateX(-50%) !important;

    width: min(72%, 1100px) !important;

    background: transparent !important;

    border: none !important;

    box-shadow: none !important;

    z-index: 9999 !important;
}




[data-testid="stChatInput"] {
    background: transparent !important;

    border: none !important;

    padding: 0 !important;
}




[data-testid="stChatInput"] > div {

    background:
        linear-gradient(
            180deg,
            #1B1B25,
            #181820
        ) !important;

    border: 1.5px solid #10F2B0 !important;

    border-radius: 24px !important;

    padding: 10px 14px !important;

    box-shadow:
        0 0 0 1px rgba(16,242,176,0.08),
        0 0 22px rgba(16,242,176,0.12);

    transition: all 0.25s ease !important;
}




[data-testid="stChatInput"] > div:hover {

    border-color: #34FFD0 !important;

    box-shadow:
        0 0 0 1px rgba(52,255,208,0.14),
        0 0 28px rgba(52,255,208,0.18);
}




[data-testid="stChatInput"] textarea {

    background: linear-gradient(
            180deg,
            #1B1B25,
            #181820
        ) !important;

    color: #F3F4F6 !important;

    border: none !important;

    font-size: 17px !important;

    padding-top: 8px !important;

    caret-color: #10F2B0 !important;

    line-height: 1.5 !important;
}




[data-testid="stChatInput"] textarea::placeholder {
    color: #9CA3AF !important;
}




textarea,
textarea:focus,
textarea:active {
    outline: none !important;
    box-shadow: none !important;
}




[data-testid="stChatInputSubmitButton"] {

    background: rgba(255,255,255,0.06) !important;

    border-radius: 14px !important;

    width: 46px !important;

    height: 46px !important;

    transition: all 0.25s ease !important;
}




[data-testid="stChatInputSubmitButton"]:hover {

    background: rgba(16,242,176,0.14) !important;

    transform: scale(1.05);
}


@media (max-width: 768px) {

    .welcome-box {

        width: 92% !important;

        padding: 45px 22px !important;

        border-radius: 30px !important;

        margin-top: 25px !important;
    }

    .welcome-box h1 {

        font-size: 34px !important;

        line-height: 1.25 !important;

        word-break: keep-all !important;
    }

    .welcome-box p {

        font-size: 18px !important;

        line-height: 1.7 !important;

        margin-top: 18px !important;
    }

    .stChatInput {

        width: 92% !important;
    }

    .user-bubble,
    .ai-bubble {

        max-width: 88% !important;

        font-size: 14px !important;
    }
}

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}
            

</style>
""", unsafe_allow_html=True)


for msg in st.session_state.messages:

    if msg["role"] == "user":

        safe_user_content = html.escape(msg["content"]).replace("\n", "<br>")

        row_html = (
            f'<div class="message-row user-row">'
            f'<div class="user-bubble">{safe_user_content}</div>'
            f'</div>'
        )

        st.markdown(row_html, unsafe_allow_html=True)

    else:

        row_html = (
            f'<div class="message-row assistant-row">'
            f'<div class="ai-bubble">{msg["content"]}</div>'
            f'</div>'
        )

        st.markdown(row_html, unsafe_allow_html=True)

if "processing" not in st.session_state:
    st.session_state.processing = False


prompt = st.chat_input("Describe your symptoms...")


if prompt and not st.session_state.processing:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    st.session_state.processing = True

    st.rerun()




if st.session_state.processing:

    last_message = st.session_state.messages[-1]["content"]

    with st.spinner("Analyzing symptoms..."):
        time.sleep(4)

   

    result = extract_symptoms(last_message)

    if result["error"]:

        reply = html.escape(result["message"])

    else:

        data = result["data"]

        symptoms = data.get("symptoms", [])

        duration = data.get("duration", 0)

        risk_result = predict_risk(
            symptoms,
            duration
        )

        prediction = risk_result["prediction"]

        rag_response = get_rag_response(last_message)

        reply = (
            f"<b>Symptoms Detected:</b> "
            f"{html.escape(', '.join(symptoms))}<br><br>"

            f"<b>Duration:</b> "
            f"{duration} day(s)<br><br>"

            f"<b>Predicted Risk:</b> "
            f"{html.escape(prediction)}<br><br>"

            f"<b>Medical Guidance:</b><br><br>"

            f"{rag_response}<br><br>"

            f"<hr>"

            f"<i>Answers provided by this AI assistant are for educational purposes only — not a substitute for professional medical advice.</i>"
        )

 

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    st.session_state.processing = False

    st.rerun()