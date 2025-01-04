import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "ChatBot with OpenAI"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant who is here to answer any questions."),
        ("user", "Question {question}"),
    ]
)


def generate_response(question, api_key, model, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=model)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"question": question})
    return response


st.title("Chatbot with OpenAI")

# Sidebar for Settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

llm = st.sidebar.selectbox(
    "Select the Language Model",
    ["gpt-4o", "o1", "chatgpt-4o-latest"],
)

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main Page
st.write("Ask me anything!")
question = st.text_input("Enter your question here")

if question:
    response = generate_response(question, api_key, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get a response.")

