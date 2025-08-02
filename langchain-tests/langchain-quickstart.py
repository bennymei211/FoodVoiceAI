import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import textwrap
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm_claude3 = ChatAnthropic(model="claude-opus-4-20250514")
llm_gpt4 = ChatOpenAI(model="gpt-4o")

# test: veryfiy LLM works
# response = llm_claude3.invoke("What is langchain?")
# print(response.content)

# basic request using system and human/user message - basic prompt engineering
# system_prompt="""
# You explain things to people like they are five year olds.
# """

# user_prompt=f"""
# What is LangChain?
# """

# messages = [
#     SystemMessage(content=system_prompt),
#     HumanMessage(content=user_prompt)
# ]

# response=llm_claude3.invoke(messages)
# answer = textwrap.fill(response.content, width=100)
# print(answer)

# CHAINS AND LOADERS
# create a simple prompt template
# prompt_template = """
# You are a helpful assistant that explains AI topics. Given the following input:
# {topic}
# Provide an explanation of the given topic"""

# # Create the prompt from the prompt template
# prompt = PromptTemplate(
#     input_variables=["topic"],
#     template=prompt_template
# )

# # Assemble the chain using the pipe operator "|", more on that later
# chain = prompt | llm_gpt4

# print(chain.invoke({"topic":"What is Langchain"}).content)

# # RUNNABLE SEQUENCE
# summarize_prompt_template = """
# You are a helpful assistant that summarizes AI concepts:
# {context}
# Summarize the context
# """

# summarize_prompt = PromptTemplate.from_template(summarize_prompt_template)

# output_parser = StrOutputParser()
# chain = summarize_prompt | llm_gpt4 | output_parser

# print(chain.invoke({"context" : "What is Langchain?"}))

# # RUNNABLELAMBDA: inject python functions into a chain with Runnable Lambda
# summarize_chain = summarize_prompt | llm_gpt4 | output_parser

# length_lambda = RunnableLambda(lambda summary: f"Summary length: {len(summary)} characters")

# lambda_chain = summarize_chain | length_lambda

# print(lambda_chain.invoke({"context" : "What is Langchain?"}))


# RUNNABLEPASSTHROUGH: for passing through unchaged or changed inputs
# unchanged is typically used with parallel
# summarize_prompt_template = """
# You are a helpful assistant that summarizes AI concepts:
# {context}
# Summarize the context
# """

# summarize_prompt = PromptTemplate.from_template(summarize_prompt_template)
# output_parser = StrOutputParser()
# summarize_chain = summarize_prompt | llm_gpt4 | output_parser

# length_lambda = RunnableLambda(lambda summary: f"Summary length: {len(summary)} characters")

# passthrough = RunnablePassthrough()

# placeholder_chain = summarize_chain | passthrough | length_lambda

# print(placeholder_chain.invoke({"context" : "What is Langchain?"}))
