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

llm_gpt4 = ChatOpenAI(model="gpt-4o")

llm_gpt4_with_structure = llm_gpt4.with_structured_output(method="json_mode")

plain_prompt_template="""
You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
{meal_input}
Provide and log nutritional values for the meal. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly.
"""

json_prompt_template="""
You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
{meal_input}
Provide and log nutritional values for the meal. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly.
Extract the information into a JSON format with this structure:

{format_structure}
"""
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

plain_prompt = PromptTemplate(
    input_variables=["meal_input"],
    template=plain_prompt_template
)

json_prompt = PromptTemplate(
    input_variables=["meal_input", "format_structure"],
    template=json_prompt_template
)

chain = plain_prompt | llm_gpt4
json_chain = json_prompt | llm_gpt4_with_structure
user_prompt = input("You: ")
print(chain.invoke({"meal_input":user_prompt}).content)
meal_as_jason = json_chain.invoke({"meal_input":user_prompt, "format_structure":format_structure})

today = date.today()
with open(f"{today}_meal_log.json", "w") as f:
    json.dump(meal_as_jason, f)