import os
from groq import Groq

# It is best practice to set your key as an environment variable
client = Groq(api_key="groq_api_key_here")

completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Explain quantum computing in one sentence."}
    ],
)

print(completion.choices[0].message.content)