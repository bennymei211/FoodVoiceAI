from openai import OpenAI
from dotenv import load_dotenv
import base64
import os
# from keys import openai_api_key

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# prompt
user_input = "Generate an image of gray tabby cat hugging an otter with an orange scarf"

response = client.images.generate(model='dall-e-3', prompt=user_input, size='1024x1024', quality='standard', n=1, style='vivid')
image_url = response.data[0].url
print(image_url)


