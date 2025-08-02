from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
# the create_stuff_documents_chain takes a list of docs and formats them all into a prompt
from langchain.chains.combine_documents import create_stuff_documents_chain

# load api keys
load_dotenv()

llm_gpt4 = ChatOpenAI(model="gpt-4o")



prompt_template="""
You are a helpful dietary assistant that logs and keeps track of meals. Given the following input:
{input}
Provide and log nutritional values for the meal. Make sure to ask for each meal. Respond in a way that is text-to-speech friendly.
"""

prompt = PromptTemplate(
    input_variables=["input"],
    template=prompt_template
)

chain = prompt | llm_gpt4
user_prompt = input("You: ")
print(chain.invoke({"input":user_prompt}).content)