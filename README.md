﻿# LLM-Gemini-Document-Reader

Imports and Setup
python
Copy code
import os
import pathlib
import sys
import textwrap
import fitz
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
This section imports required libraries:

Standard libraries (os, pathlib, sys, etc.) for file handling and system operations.
Libraries for AI tasks, including langchain, google.generativeai, sentence-transformers, and numpy.
Text processing libraries, such as textwrap for formatting, and fitz for handling PDF documents.
Other utilities, including IPython for interactive environments and dotenv for managing environment variables.
Google Generative AI Configuration
python
Copy code
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
Sets the Google API key for accessing Generative AI services from environment variables using dotenv.
Markdown Formatting Function
python
Copy code
def to_markdown(text):
    text = text.replace('.', ' *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
Converts text into a Markdown-formatted string with a specific style for display in notebooks.
Model Selection
python
Copy code
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
Lists and filters models capable of content generation, printing their names.
Prompt and Output Parser Initialization
python
Copy code
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent chatbot. Answer the following question."),
        ("user", "{question}")
    ]
)
Initializes an output parser and defines a system and user prompt structure for interacting with the LLM.
PDF Loading
python
Copy code
loader = PyPDFLoader(r"C:\Users\ISHAN\Music\Coronavirus.pdf")
all_contents = loader.load()
Loads a PDF file from a given path using PyPDFLoader.
Text Splitting
python
Copy code
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=60)
chunks = text_splitter.split_documents(all_contents)
Splits the loaded PDF into chunks of text (1,000 characters with a 60-character overlap) for better processing.
Embedding and Vector Store Initialization
python
Copy code
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector = embeddings_model.embed_query(str(chunks))

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
Converts text chunks into vector embeddings using the specified Generative AI embedding model.
Stores these embeddings in a Chroma vector database.
Retriever Initialization
python
Copy code
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 10})
Sets up a retriever for similarity-based search, fetching the top 10 matching documents.
System Prompt
python
Copy code
system_prompt = (
    "You are an intelligent chatbot that answers only using the information provided in the loaded data."
    " If the loaded data does not contain any relevant information for the question, respond followed by a general knowledge answer."
    " Do not blend the loaded data with general knowledge when answering."
    "\n\n"
    "responses need to short for only 20 words"
    "Context: {context}"
    "\n"
    "Note: Provide a general knowledge answer if the loaded data does not contain the required information, but clearly separate the two responses."
)
Defines a detailed system prompt for the LLM to generate context-aware responses.
Question-Answer Function
python
Copy code
def qa(question):
    # Build the history string
    history_str = ""
    for h in history:
        history_str += f"q: {h['question']}, a: {h['answer']}\n"

    # Generate the prompt text
    prompt_text = prompt.format(input=question, context="\n\n".join([
                                doc.page_content for doc in chunks]))

    # Generate the response from the model
    response = llm.generate_content(prompt_text)  # Using the available method

    # Access the text content from the response object
    response_text = response.text.strip()  # Trim any leading/trailing whitespace

    # Clean up the response text to remove unwanted newline characters
    response_text = response_text.replace("\n", " ")

    # Extract the answer based on common response patterns
    if "no idea" in response_text.lower():
        # Get the response after "Chatbot: "
        final_answer = response_text.split("Chatbot: ")[-1].strip()
    else:
        # Return the full response if "no idea" is not found
        final_answer = response_text.strip()

    # Update history with the new question and extracted answer
    history.append({"question": question, "answer": final_answer})

    return final_answer
Implements the QA logic:
Formats the history of interactions.
Sends a formatted prompt to the LLM.
Processes the response to extract the relevant answer.
PDF QA Workflow
python
Copy code
def load_pdf_and_answer(question):
    return qa(question)
Wrapper function to load a PDF and get answers to a given question.
Model Wrapper Class
python
Copy code
class LLMWrapper(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, prompt):
        response = self.llm.generate_content(prompt)
        return response.text
Wraps the LLM functionality into a class, standardizing how prompts are sent and responses retrieved.
RAG Chain Setup
python
Copy code
qa_chain = create_stuff_documents_chain(llm_wrapper, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)
Combines the retriever with a question-answer chain into a Retrieval-Augmented Generation (RAG) pipeline for document-based QA.
