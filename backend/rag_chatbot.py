from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os



load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY)



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)



def get_rag_response(user_query):

    try:

        

        retrieved_docs = retriever.invoke(user_query)

        if not retrieved_docs:
            return (
                "I could not find enough medical "
                "information for this query."
            )

    

        context = "\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )

 
        prompt = f"""

You are a medically cautious AI healthcare assistant.

If the user's message is casual, general, or non-medical:
- Respond naturally like a friendly assistant.
- DO NOT explain that the query is non-medical.
- DO NOT mention classification, analysis, symptoms, risks, or reasoning.
- Simply continue the conversation normally.
- Optionally ask if they need medical assistance.

- Only provide medical analysis if the user clearly mentions symptoms, diseases, health concerns, medications, or medical questions.

- Answer medical questions ONLY using the provided medical context.

- If the context does not contain enough information, say:
  "I do not have enough medical information to answer safely."

- Do not invent diseases, treatments, medications, risks, or symptoms.

Medical Context:
{context}

User Question:
{user_query}

If it is a medical query, provide:
1. Possible condition
2. Basic precautions
3. Whether doctor consultation is needed
4. Emergency warning signs if any

Keep the response concise, medically safe, and easy to understand.
"""
    

        response = client.chat.completions.create(
            model="groq/compound",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        return answer

    except Exception as e:

        return f"unable to process the query due to an error. Plz try again later"