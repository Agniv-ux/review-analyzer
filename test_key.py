import os, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.getenv("LLM_API_KEY") or ""
print("Key found:", "yes" if key and key != "paste-your-key-here" else "NO")
print("Model:", os.getenv("LLM_MODEL"))
print("Asking the AI one small question (waits 30 seconds at most)...")

client = OpenAI(api_key=key or "none", base_url=os.getenv("LLM_BASE_URL"), max_retries=0, timeout=30)
start = time.time()
try:
    reply = client.chat.completions.create(
        model=os.getenv("LLM_MODEL") or "none",
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print("AI replied:", reply.choices[0].message.content)
except Exception as error:
    print("ERROR:", type(error).__name__, error)
print(f"Took {time.time() - start:.1f} seconds")
