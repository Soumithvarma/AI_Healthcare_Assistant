

import json
import re
import os

from groq import Groq
from dotenv import load_dotenv



load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY)




def extract_symptoms(user_input):

    prompt = f"""
You are a medical symptom extraction assistant.

Extract:
- symptoms
- duration

Return ONLY valid JSON.

Rules:
1. No explanation
2. No markdown
3. Symptoms must be concise
4. No hallucinated symptoms or details
5. If a symptom contains multiple words, replace spaces with underscores.
6. Convert duration into integer number of days.
7. morning, afternoon, evening, today, yesterday = 1 day.

Example:
{{
    "symptoms": ["fever", "vomiting", "chest_pain"],
    "duration": 2
}}

User Text:
"{user_input}"
"""

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        raw_text = response.choices[0].message.content.strip()

     
        raw_text = re.sub(r"```json|```", "", raw_text).strip()

        data = json.loads(raw_text)

        return {
            "error": False,
            "data": data
        }

    except Exception as e:

        print("API Error:", e)

        return {
            "error": True,
            "message": "Unable to analyze symptoms right now. Please try again later."
        }