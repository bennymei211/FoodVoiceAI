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
from langchain_community.document_loaders import JSONLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder




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

# returns gpt-4o response as a json-formatted string
def get_gpt_json_response(llm_with_structure, user_input):
    # json format prompt template
    json_prompt_template="""
    You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
    {meal_input}
    Provide and log nutritional values for the meal. 
    Extract the information into a JSON format with this structure:

    {format_structure}
    """

    # # structure for json format
    # format_structure = """{
    #     "meal": "...",sssssasa
    #     "dish_name": "...",
    #     "ingredients": ["...", "..."],
    #     "nutritional_info": {
    #         "protein": "...",
    #         "fat": "...",
    #         "carbs": "...",
    #         "sodium": "...",
    #         ...
    #     }
    # }"""

    class NutritionalInfo(BaseModel):
        protein: float = Field(description="Amount of protein in grams")
        fat: float = Field(description="Amount of fat in grams")
        carbohydrates: float = Field(description="Amount of carbohydrates in grams")
        sodium: float = Field(description="Amount of sodium in milligrams")
        fiber: float = Field(description="Amount of fiber in grams")
        sugar: float = Field(description="Amount of sugar in grams")

    class Meal(BaseModel):
        meal: str = Field(..., description="Meal of the day")
        dish_name: str = Field(description="Name of the dish")
        ingredients: List[str] = Field(description="List of ingredients from the dish")
        nutritional_info: NutritionalInfo = Field(description="Nutritional information for the dish")



    # prompt for json format
    json_prompt = ChatPromptTemplate.from_template(json_prompt_template)

    parser = JsonOutputParser(pydantic_object=Meal)

    # chain for json format generation
    json_chain = json_prompt | llm_with_structure | parser

    meal_as_json = json_chain.invoke({"meal_input":user_input, "format_structure":parser.get_format_instructions()})
    
    return meal_as_json

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

def speak(text, engine):
    engine.say(text)
    engine.runAndWait()

def append_json_entry(new_entry, filename="data.json"):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump([], f, indent=4)

    with open(filename, 'r+') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data =[]

        data.append(new_entry)

        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()

if __name__ == "__main__":
    # load api keys
    load_dotenv()

    # initialize pyttsx3 engine
    engine = pyttsx3.init()
    # initialize speech recognizer
    recognizer = sr.Recognizer()

    # wrapper for gpt-4o plain conversation generation and gpt-4o json format generation
    llm_gpt4 = ChatOpenAI(model="gpt-4o")

    chat_history = [
        AIMessage(content='What did you have for breakfast?')
    ]


    flag = 0
    while True:
        try:
            engine.say("...")
            if flag == 0:
                print("ChatBot: Hello! What did you have for breakfast?")
                speak("Hello! What did you have for breakfast?", engine)
                time.sleep(1.5)
            flag = 1
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                # get user input
                # user_prompt = input("You: ")
                audio = recognizer.listen(source)
                user_prompt = recognizer.recognize_google(audio)
                print("You:", user_prompt)

                # exit conition
                if user_prompt.lower() in ["quit", "exit"]:
                    speak("Thanks for sharing! Goodbye!", engine)
                    print("ChatBot: Thanks for sharing! Goodbye!")
                    time.sleep(4)
                    break

                response = get_gpt_response(llm=llm_gpt4, user_input=user_prompt, chat_history=chat_history)
                chat_history.append(HumanMessage(content=user_prompt))
                chat_history.append(AIMessage(content=response))
                image_prompt = get_dalle_prompt(user_input=user_prompt)
                image_url = get_dalle3_image(prompt=user_prompt)
                print("ChatBot:", response)
                print(image_url)
                speak(response, engine)
                time.sleep(1.5)

                # output to json file using today's date
                today = date.today()
                append_json_entry(new_entry=get_gpt_json_response(llm_with_structure=llm_gpt4, user_input=user_prompt), filename=f"{today}_meal_log.json")
        except sr.UnknownValueError:
            time.sleep(4)
            print("Sorry, I couldn't understand the audio")
            speak("Sorry, I couldn't understand the audio", engine)
            time.sleep(1.5)
    engine.stop()

