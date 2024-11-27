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

if 'google.colab' in sys.modules:
    from google.colab import userdata
else:
    print("Not running in Google Colab, skipping 'google.colab' import.")
import os
import textwrap

import fitz
import google.generativeai as genai
import langchain_google_genai
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from IPython.display import Markdown, display
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()  # This loads the environment variables from the .env file

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def to_markdown(text):
    text = text.replace('.', ' *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

# Initialize the string output parser
parser = StrOutputParser()

# Create a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent chatbot. Answer the following question."),
        ("user", "{question}")
    ]
)

llm = genai.GenerativeModel(
    model_name='models/gemini-pro',
    generation_config={},
    safety_settings={},
    tools=None,
    system_instruction=None,

)


# Use a raw string for the file path to avoid issues with backslashes
loader = PyPDFLoader(r"C:\Users\ISHAN\Music\Coronavirus.pdf")

# Load the PDF data
all_contents = loader.load()

# Initialize the text splitter with the desired chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=60)

# Split the loaded PDF data into chunks
chunks = text_splitter.split_documents(all_contents)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004")
vector = embeddings_model.embed_query(str(chunks))

vectorstore = Chroma.from_documents(
    documents=chunks, embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))


retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 10})

system_prompt = (
    "You are an intelligent chatbot that answers only using the information provided in the loaded data."
    " If the loaded data does not contain any relevant information for the question, respond  followed by a general knowledge answer."
    " Do not blend the loaded data with general knowledge when answering."
    "\n\n"
    "responses need to short for only 20 words"
    "Context: {context}"
    "\n"
    "Note: Provide a general knowledge answer if the loaded data does not contain the required information, but clearly separate the two responses."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

history = []


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


def load_pdf_and_answer(question):
    # Call the qa() function
    return qa(question)


# Ask a question
question = ""
answer = load_pdf_and_answer(question)

# Print the answer
print("Answer:", answer)


class LLMWrapper(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, prompt):
        response = self.llm.generate_content(prompt)
        return response.text  # Adjust according to the actual attribute


# Instantiate the wrapper with your model
llm_wrapper = LLMWrapper(llm)

# Now, use the wrapper when creating the chain
qa_chain = create_stuff_documents_chain(llm_wrapper, prompt)

# Create the RAG chain
rag_chain = create_retrieval_chain(retriever, qa_chain)
