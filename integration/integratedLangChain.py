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
import asyncio
import time
from dotenv import load_dotenv
from datetime import date
from typing import List
from pathlib import Path
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
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
client = OpenAI()

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

def download_image_to_temp(url):
    #This function downloads an image from the internet.Loads it into memory and saves it as a temporary .png file.
    #Returns both the temporary file path and the in-memory image object.
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(temp_file.name)
    return temp_file.name, image

# def get_refined_prompt(image_path, instruction):
#     #This function reads and Base64-encodes the image and sends it along with your instructionto gpt-4o.
#     #Then the model processes both the text and image and returns the model's refined descriptionor instructions for that image.
#     with open(image_path, "rb") as img_file:
#         base64_img = base64.b64encode(img_file.read()).decode("utf-8")

#     response = openai.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": f"Refine this image using: {instruction}"},
#                     {"type": "image_url", "image_url": {
#                         "url": f"data:image/png;base64,{base64_img}",
#                         "detail": "high"
#                     }}
#                 ]
#             }
#         ],
#         max_tokens=300
#     )
#     return response.choices[0].message.content.strip()

def langchain_get_refined_prompt(llm, instruction, chat_history, image_path):
    with open(image_path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode("utf-8")
    
    # plain conversation prompt template
    system_message="""
    You are a helpful image refinement assistant.
    Given an image, context, and instruction, refine or describe the most recent dish's image according to the instruction.
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

def get_gpt_image1_prompt(user_input):
    image_prompt_template="""
    Create a DALL-E-3 image prompt describing a photorealistic, top-down view of ONLY the following meal:
    {meal_input}
    - The background must be plain white.
    - Do NOT include any utensils, decorations, or extra items.
    - Focus only on the lsited food items
    - No table or background context should appear
    - The food should be centered
    """

    image_prompt = ChatPromptTemplate.from_template(image_prompt_template)
    return image_prompt.format(meal_input=user_input)

def get_gpt_image1_image(prompt, filename = "food.png"):
    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
        tools=[
            {
                "type": "image_generation",
                "size": "1024x1024",
                "quality": "low", 
            }
        ],
    )

    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    if image_data:
        image_base64 = image_data[0]

        with open(filename, "wb") as f:
            f.write(base64.b64decode(image_base64))

    return response
    
def refine_gpt_image1_image(instruction, previous_response_id, filename = "refined_food.png"):
    refined_response = client.responses.create(
        model="gpt-4o",
        previous_response_id=previous_response_id,
        input=instruction,
        tools=[
            {
                "type": "image_generation",
                "size": "1024x1024",
                "quality": "low", 
            }
        ],
    )

    image_data = [
        output.result
        for output in refined_response.output
        if output.type == "image_generation_call"
    ]

    if image_data:
        image_base64 = image_data[0]
        with open(filename, "wb") as f:
            f.write(base64.b64decode(image_base64))

    return refined_response

def tts(input: str) -> None:
    speech_file_path = Path(__file__).parent / "speech.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=input,
        response_format="mp3",
    ) as response:
        response.stream_to_file(speech_file_path)

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64, {b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(md, unsafe_allow_html=True)
    


# Streamlit UI starts here
st.set_page_config(layout="wide")
st.title("🧠 GPT-4o + GPT-Image-1 | Conversational Image Refiner")

# Session state
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "step" not in st.session_state:
    st.session_state.step = 0
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []
if "audio_generated" not in st.session_state:
    st.session_state.audio_generated = False;

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
            image_prompt = get_gpt_image1_prompt(initial_prompt)
            image_response = get_gpt_image1_image(image_prompt, filename="initial_food.png")
            text_response = get_gpt_response(llm, initial_prompt, st.session_state.prompt_history)

            st.session_state.previous_response_id = image_response.id
            st.session_state.image_path = "initial_food.png"
            st.session_state.step = 1
            st.session_state.prompt_history.append(HumanMessage(content=initial_prompt))
            st.session_state.prompt_history.append(AIMessage(content=text_response))
            tts(text_response)
            st.session_state.audio_generated = True
            st.rerun()

            # image_prompt = get_image_prompt(initial_prompt)
            # url = get_dalle3_image(image_prompt)
            # path, img = download_image_to_temp(url)
            # st.session_state.image_path = path
            # st.session_state.step = 1
            # st.rerun()

    else:
        refine_prompt = st.text_area("Refine the current image", height=100)
        if st.button("Refine Image"):
            refined_image_prompt = get_gpt_image1_prompt(refine_prompt)
            refined_image_response = refine_gpt_image1_image(
                refined_image_prompt, 
                st.session_state.previous_response_id, 
                filename="refined_food.png"
            )
            refined_text_response = get_gpt_response(llm, refine_prompt, st.session_state.prompt_history)

            st.session_state.previous_response_id = refined_image_response.id
            st.session_state.image_path = "refined_food.png"
            st.session_state.step += 1
            st.session_state.prompt_history.append(HumanMessage(content=refine_prompt))
            st.session_state.prompt_history.append(AIMessage(content=refined_text_response))
            tts(refined_text_response)
            st.session_state.audio_generated = True
            st.rerun()


            # refined_text = langchain_get_refined_prompt(llm, refine_prompt, st.session_state.prompt_history, st.session_state.image_path) #(llm, instruction, chat_history, image_path):
            # image_prompt = get_image_prompt(refined_text)
            # url = get_dalle3_image(image_prompt)
            # os.remove(st.session_state.image_path)
            # path, img = download_image_to_temp(url)
            # st.session_state.image_path = path
            # st.session_state.step += 1
            # st.session_state.prompt_history.append((refine_prompt, refined_text))
            # st.rerun()
    
    if st.session_state.get("audio_generated", False):
        autoplay_audio("speech.mp3")
        st.session_state.audio_generated = False

    if st.button("Reset"):
        st.session_state.image_path = None
        st.session_state.step = 0
        st.session_state.prompt_history = []
        st.rerun()

    # Show previous prompts
    if st.session_state.prompt_history:
        st.markdown("### 🧾 History")
        for msg in st.session_state.prompt_history[::-1]:
            if isinstance(msg, HumanMessage):
                st.markdown(f"**You:** {msg.content}")
            else:
                st.markdown(f"**GPT:** {msg.content}")
