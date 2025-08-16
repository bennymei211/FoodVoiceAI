import streamlit as st
import openai
import requests
from PIL import Image
from io import BytesIO
import base64
import os
import tempfile

from dotenv import load_dotenv
import json
import pyttsx3
import os
import time
from dotenv import load_dotenv
from datetime import date
from typing import List
from openai import OpenAI
import speech_recognition as sr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field
# from langchain_community.document_loaders import JSONLoader
# from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder


load_dotenv()

# image prompt for dall-e-3
def get_dalle_prompt(user_input):
    image_prompt_template="""
    Create a DALL-E-3 image prompt describing a photorealistic, top-down view of ONLY the following meal:
    {meal_input}
    - The background must be plain white.
    - Do NOT include any plates, utensils, decorations, or extra items.
    - Focus only on the lsited food items and their ingredients
    - No table or background context should appear
    - The food should be centered
    """

    image_prompt = ChatPromptTemplate.from_template(image_prompt_template)
    return image_prompt.format(meal_input=user_input)

def get_dalle3_image(prompt):
    client = OpenAI()
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
        response_format="url"
    )
    return response.data[0].url

# prints response from gpt-4o
def get_gpt_response(llm, user_input, chat_history):
    # plain conversation prompt template
    system_message="""
    You are a helpful dietary assistant that logs and keeps track of meals. Make sure to ask for each meal. Given the following input:
    {meal_input}
    Provide and log nutritional values for the meal. Respond in a way that is text-to-speech friendly.
    """

    # prompt for plain conversation
    plain_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{meal_input}")
    ])

    # chain for plain conversation generation
    chain = plain_prompt | llm

    response = chain.invoke({"meal_input":user_input, "chat_history": chat_history}).content
    return response

# # Set your OpenAI API key
# def generate_image(prompt):
#     #This function uses DALLE.E 3 model to create an image from a text prompt and returns the URL where the generated image can be accessed.
#     response = openai.images.generate(
#         model="dall-e-3",
#         prompt=prompt,
#         size="1024x1024",
#         quality="standard",
#         n=1,
#         response_format="url"
#     )
#     return response.data[0].url

def download_image_to_temp(url):
    #This function downloads an image from the internet.Loads it into memory and saves it as a temporary .png file.
    #Returns both the temporary file path and the in-memory image object.
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_file.name)
    return temp_file.name, image

def get_refined_prompt(image_path, instruction):
    #This function reads and Base64-encodes the image and sends it along with your instructionto gpt-4o.
    #Then the model processes both the text and image and returns the model's refined descriptionor instructions for that image.
    with open(image_path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Refine this image using: {instruction}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_img}",
                        "detail": "high"
                    }}
                ]
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def langchain_get_refined_prompt(llm, instruction, chat_history, image_path):
    with open(image_path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode("utf-8")
    
    # plain conversation prompt template
    system_message="""
    You are a helpful image refinement assistant.
    Given an image and instruction, refine or describe the image according to the instruction.
    """

    # prompt for plain conversation
    plain_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", [
            {"type": "text", "text": f"Refine this image using: {instruction}"},
            {"type": "image_url", "image_url": f"data:image/png;base64,{base64_img}"}
        ])
    ])

    # chain for plain conversation generation
    chain = plain_prompt | llm

    response = chain.invoke({"chat_history": chat_history}).content
    return response



# Streamlit UI starts here
st.set_page_config(layout="wide")
st.title("🧠 GPT-4o + DALL·E 3 | Conversational Image Refiner")

# Session state
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "step" not in st.session_state:
    st.session_state.step = 0
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

# Layout: 2 columns
col1, col2 = st.columns([2, 1])

# Left: Show Image
with col1:
    if st.session_state.image_path:
        st.image(st.session_state.image_path, use_container_width=True, caption=f"Step {st.session_state.step}")

# Right: Prompt Input
with col2:
    llm = ChatOpenAI(model="gpt-4o")
    st.subheader("💬 Describe your prompt")
    if st.session_state.step == 0:
        initial_prompt = st.text_input("Initial prompt", value="chicken biryani on a plate", key="initial")
        if st.button("Generate Initial Image"):
            image_prompt = get_dalle_prompt(initial_prompt)
            url = get_dalle3_image(image_prompt)
            path, img = download_image_to_temp(url)
            st.session_state.image_path = path
            st.session_state.step = 1
            st.rerun()

    else:
        refine_prompt = st.text_area("Refine the current image", height=100)
        if st.button("Refine Image"):
            refined_text = langchain_get_refined_prompt(llm, refine_prompt, st.session_state.prompt_history, st.session_state.image_path) #(llm, instruction, chat_history, image_path):
            image_prompt = get_dalle_prompt(refined_text)
            url = get_dalle3_image(image_prompt)
            os.remove(st.session_state.image_path)
            path, img = download_image_to_temp(url)
            st.session_state.image_path = path
            st.session_state.step += 1
            st.session_state.prompt_history.append((refine_prompt, refined_text))
            st.rerun()

    if st.button("Reset"):
        st.session_state.image_path = None
        st.session_state.step = 0
        st.session_state.prompt_history = []
        st.rerun()

    # Show previous prompts
    if st.session_state.prompt_history:
        st.markdown("### 🧾 History")
        for i, (u, g) in enumerate(st.session_state.prompt_history[::-1], 1):
            st.markdown(f"**You:** {u}\n\n**GPT:** {g}")
