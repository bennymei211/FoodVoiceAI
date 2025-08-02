import json
from dotenv import load_dotenv
from datetime import date
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
# the create_stuff_documents_chain takes a list of docs and formats them all into a prompt
from langchain.chains.combine_documents import create_stuff_documents_chain

# load api keys
load_dotenv()

# wrapper for gpt-4o plain conversation generation
llm_gpt4 = ChatOpenAI(model="gpt-4o")

# wrapper for gpt-4o json format generation
llm_gpt4_with_structure = llm_gpt4.with_structured_output(method="json_mode")

# plain conversation prompt template
plain_prompt_template="""
You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
{meal_input}
Provide and log nutritional values for the meal. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly.
"""

# json format prompt template
json_prompt_template="""
You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
{meal_input}
Provide and log nutritional values for the meal. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly.
Extract the information into a JSON format with this structure:

{format_structure}
"""

# structure for json format
format_structure = """{
    "meal": "...",
    "dish_name": "...",
    "ingredients": ["...", "..."],
    "nutritional_info": {
        "protein": "...",
        "fat": "...",
        "carbs": "...",
        "sodium": "...",
        ...
    }
}"""

# prompt for plain conversation
plain_prompt = PromptTemplate(
    input_variables=["meal_input"],
    template=plain_prompt_template
)

# prompt for json format
json_prompt = PromptTemplate(
    input_variables=["meal_input", "format_structure"],
    template=json_prompt_template
)

# chain for plain conversation generation
chain = plain_prompt | llm_gpt4

# chain for json format generation
json_chain = json_prompt | llm_gpt4_with_structure

# get user input
user_prompt = input("You: ")

print(chain.invoke({"meal_input":user_prompt}).content)
meal_as_json = json_chain.invoke({"meal_input":user_prompt, "format_structure":format_structure})

# output to json file using today's date
today = date.today()
with open(f"{today}_meal_log.json", "w") as f:
    json.dump(meal_as_json, f)

def get_gpt_response(llm, user_input):
    # plain conversation prompt template
    plain_prompt_template="""
    You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
    {meal_input}
    Provide and log nutritional values for the meal. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly.
    """

    # prompt for plain conversation
    plain_prompt = PromptTemplate(
        input_variables=["meal_input"],
        template=plain_prompt_template
    )

    # chain for plain conversation generation
    chain = plain_prompt | llm

    print(chain.invoke({"meal_input":user_input}).content)

def get_gpt_json_response(llm_with_structure, user_input):
    # json format prompt template
    json_prompt_template="""
    You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
    {meal_input}
    Provide and log nutritional values for the meal. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly.
    Extract the information into a JSON format with this structure:

    {format_structure}
    """

    # structure for json format
    format_structure = """{
        "meal": "...",
        "dish_name": "...",
        "ingredients": ["...", "..."],
        "nutritional_info": {
            "protein": "...",
            "fat": "...",
            "carbs": "...",
            "sodium": "...",
            ...
        }
    }"""

    # prompt for json format
    json_prompt = PromptTemplate(
        input_variables=["meal_input", "format_structure"],
        template=json_prompt_template
    )

    # chain for json format generation
    json_chain = json_prompt | llm_with_structure

    meal_as_json = json_chain.invoke({"meal_input":user_input, "format_structure":format_structure})
    
    return meal_as_json


