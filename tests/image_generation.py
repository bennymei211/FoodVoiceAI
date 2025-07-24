from openai import OpenAI
from dotenv import load_dotenv
import os
# from keys import openai_api_key

client = OpenAI(api_key="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")

# prompt
user_input = "Generate an image of gray tabby cat hugging an otter with an orange scarf"

response = client.images.generate(model='dall-e-3', prompt=user_input, size='1024x1024', quality='standard', n=1, style='vivid')
image_url = response.data[0].url
print(image_url)