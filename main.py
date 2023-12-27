import os
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv, dotenv_values
from pprint import pprint 
from lc_func import langchainChat

config = dotenv_values(".env")

API_KEY = config["OPENAI_API_KEY"]
PINE_API_KEY = config["PINE_API_KEY"]
EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_NAME = "llama-2-rag"

pinecone.init(
        api_key=PINE_API_KEY
    )

chat = ChatOpenAI(
    openai_api_key=API_KEY,
    model='gpt-3.5-turbo'
)

embed_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=API_KEY)

text_field = "text"

index = pinecone.Index(INDEX_NAME)

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

chat = langchainChat(chat, vectorstore)
chat.chat_start()