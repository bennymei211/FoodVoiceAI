import json
import pyttsx3
from dotenv import load_dotenv
from datetime import date
from typing import List
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_community.document_loaders import JSONLoader
from langchain_core.runnables.history import RunnableWithMessageHistory


# prints response from gpt-4o
def get_gpt_response(llm, user_input):
    # plain conversation prompt template
    plain_prompt_template="""
    You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
    {meal_input}
    Provide and log nutritional values for the meal. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly.
    """

    # prompt for plain conversation
    plain_prompt = ChatPromptTemplate.from_template(plain_prompt_template)

    # chain for plain conversation generation
    chain = plain_prompt | llm

    response = chain.invoke({"meal_input":user_input}).content
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
    #     "meal": "...",
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
    Create a DALL-E-3 prompt that describes a photorealistic image of the following meal:
    {meal_input}
    with a plain white background, no decorations or extra items, and only include the exact food items listed.

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

if __name__ == "__main__":
    # load api keys
    load_dotenv()

    # initialize pyttsx3 engine
    engine = pyttsx3.init()
    print("ChatBot: Hello! What did you have for breakfast?")
    # speak("Hello! What did you have for breakfast?", engine)

    # wrapper for gpt-4o plain conversation generation and gpt-4o json format generation
    llm_gpt4 = ChatOpenAI(model="gpt-4o")

    while True:
        # get user input
        user_prompt = input("You: ")

        # exit conition
        if user_prompt.lower() in ["quit", "exit"]:
            break

        response = get_gpt_response(llm=llm_gpt4, user_input=user_prompt)
        # image_prompt = get_dalle_prompt(user_input=user_prompt)
        # image_url = get_dalle3_image(prompt=user_prompt)
        print(response)
        # print(image_url)

        # output to json file using today's date
        today = date.today()
        with open(f"{today}_meal_log.json", "w") as f:
            json.dump(get_gpt_json_response(llm_with_structure=llm_gpt4, user_input=user_prompt), f)
    
    engine.stop()

