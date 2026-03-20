from google import genai

client = genai.Client(api_key="AIzaSyDt9U2Q9gximSv__tbcV8RmKway0_PSQWs")

response = client.models.generate_content(
    model="gemini-2.5-flash",   # ← change here
    contents="Hello, how are you?"
)

print(response.text)
