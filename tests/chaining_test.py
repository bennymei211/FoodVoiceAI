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

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


load_dotenv(find_dotenv())

# ----llm wrapping----
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# ----prompt template----
# Prompt1: expert explanation
prompt1 = PromptTemplate.from_template("""
You are an expert data scientist with an expertise in building deep learning models.
Explain the concept of {concept} in a couple of lines.
"""
)

# Prompt2: simplified explanation
prompt2 = PromptTemplate.from_template(
    "Turn the concept description of {ml_concept} and explain it to me like I'm five in 500 words"
)

# ----chaining----
# chain1: Takes "concept" and produces a response
chain1 = prompt1 | llm

# chain2: Takes the result of chain1 and formats it as input for the next prompt
chain2 = prompt2 | llm

# overall_chain: pipe output from chain1 as "concept" into chain2
overall_chain = RunnableSequence(
    first=chain1,
    last=(lambda output: {"ml_concept": output}) | chain2
)

result = overall_chain.invoke({"concept": "autoencoder"})
# print(result.content)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0
)

texts = text_splitter.create_documents([result.content])
# print(texts)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
query_result = embeddings.embed_query(texts[0].page_content)
print(query_result)