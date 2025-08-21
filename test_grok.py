import os
from openai import OpenAI

# Set API key (in real use, load from env or .env)
api_key = "xai-wcEpsWH7JHqUV0ZA2KwuD9rXY8GJO6qseSJAsNZitmo1oSZGPFK64JnAHHUqnK2GPAFA5DtAhNJUMiiC"

client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1"
)

try:
    response = client.completions.create(
        model="grok-beta",
        prompt="Say hello from Grok!",
        max_tokens=10
    )
    print("Response:", response.choices[0].text.strip())
except Exception as e:
    print("Error:", str(e))