from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_completion(prompt, model="llama-3.1-8b-instant", temperature=0.1):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer strictly from context."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content
