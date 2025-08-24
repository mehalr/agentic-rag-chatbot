# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

# import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
import asyncio
import nest_asyncio

# load environment variables
load_dotenv()  


# initialize embeddings and vector store in a function
def init_embeddings_and_store():
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()

    # Run in async context
    async def init_async():
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Run the async initialization in the event loop
    embeddings = asyncio.run(init_async())

    # initiating vector store
    return embeddings, Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

# Initialize embeddings and vector store
if 'embeddings' not in st.session_state:
    st.session_state.embeddings, st.session_state.vector_store = init_embeddings_and_store()
 
# initiating llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)

# Create the agent prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a helpful AI assistant that uses the provided tools to answer questions.
    You have access to a tool that can retrieve information from a knowledge base.
    Always try to provide accurate information based on the retrieved content."""),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Load and process documents only once when initializing the app
if 'documents_loaded' not in st.session_state:
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = st.session_state.vector_store.add_documents(documents=all_splits)
    st.session_state.documents_loaded = True


# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = st.session_state.vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("How are you?")


# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))