from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # required by the client but not used by Ollama
)

MODEL = "gemma3:1b"


def get_llm_response(user_text):
    print("🤖 Thinking...")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a voice assistant. Your response will be read aloud by a text-to-speech engine. "
                    "You MUST follow these rules strictly:\n"
                    "1. Write in plain sentences only. No bullet points, no numbered lists, no headers.\n"
                    "2. Do NOT use any markdown: no *, no **, no #, no -, no backticks.\n"
                    "3. Keep your answer to 2-3 sentences maximum.\n"
                    "4. Sound natural and conversational, as if speaking to someone face to face.\n"
                    "5. Never start your response with 'Sure', 'Certainly', 'Of course', or similar filler."
                )
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        temperature=0.7,       # balanced creativity vs focus
        max_tokens=150,        # cap response length for voice output
    )

    return response.choices[0].message.content.strip()
