from google import genai

client = genai.Client(api_key="AIzaSyDt9U2Q9gximSv__tbcV8RmKway0_PSQWs")

for m in client.models.list():
    print(m.name)
