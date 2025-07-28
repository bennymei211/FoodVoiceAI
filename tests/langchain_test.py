from dotenv import load_dotenv, find_dotenv
# llm wrapping imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
# prompt imports
from langchain_core.prompts import PromptTemplate
# chain imports
from langchain_core.runnables import RunnableSequence

load_dotenv(find_dotenv())

# ----llm wrapping----
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
llm.invoke("explain large language models in one sentence")

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a Python script that trains a neural network on simulated data")
]
response = chat.invoke(messages)
print(response.content, end='\n')




# ----prompt template----
template = """"
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {concept} in a couple of lines
"""

prompt = PromptTemplate(
    input_variables=["concept"],
    template=template
)

print(prompt, "\n")
print(llm.invoke(prompt.format(concept="regularization")).content)





