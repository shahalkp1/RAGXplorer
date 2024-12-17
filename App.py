import streamlit as st
import os
from pydantic_ai import Agent, RunContext
from pathlib import Path
from pydantic_ai.models.groq import GroqModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nest_asyncio
from dataclasses import dataclass
from tavily import TavilyClient
from typing import List
from typing import Union
import tempfile
import json


nest_asyncio.apply()
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')
loaders,  documents = [], []
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

groq_model = GroqModel("llama-3.3-70b-versatile")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def split_document(loader):
    document = loader.load()
    split_docs = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50).split_documents(document)
    return split_docs

def create_vs(docs):
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectorstore.persist()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 7})
    return retriever

@dataclass
class Deps:
    question: Union[str, None]
    context: Union[str, None]

groq_agent = Agent(groq_model,
                   deps_type=Deps,
                    retries=2,
                    result_type=str,
                   system_prompt=("You are a Helpful Assiatnt Profiocient in Answering concise,factful and to the point asnwers for questions asked based on the Context provided"
                   "You have to Use the `retrievre_tool' to get relevent context and generate response based on the context retrieved"
                   """You are a grading assistant. Evaluate the response based on:
        1. Relevancy to the question
        2. Faithfulness to the context
        3. Context quality and completeness
        
        lease grade the following response based on:
        1. Relevancy (0-1): How well does it answer the question?
        2. Faithfulness (0-1): How well does it stick to the provided context?
        3. Context Quality (0-1): How complete and relevant is the provided context?
        
        Question: {ctx.deps.query}
        Context: {ctx.deps.context}
        Response: {ctx.deps.response}
        
        Also determine if web search is needed to augment the context.
        
        Provide the grades and explanation in the JSON format with key atrributes 'Relevancy','Faithfulness','Context Quality','Needs Web Search':
        {"Relevancy": <score>,
        "Faithfulness": <score>,
        "Context Quality": <score>,
        "Needs Web Search": <true/false>,
        "Explanation": <explanation>,
        "Answer":<provide response based on the context from the `retrievre_tool' if 'Need Web Search' value is 'false' otherwise Use the `websearch_tool` function to generate the final reaponse}"""
        ),
        )

@groq_agent.tool_plain
async def websearch_tool(question) -> str:  
    tavily_client = TavilyClient(api_key="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")
    answer = tavily_client.qna_search(query=question)
    return answer

@groq_agent.tool
async def retriever_tool(ctx: RunContext[Deps], question: str) -> List[str]:
    load_vectorstore = Chroma(
        persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix(), 
        embedding_function=embedding
    )
    docs = load_vectorstore.similarity_search(question, k=3)
    return [d.page_content for d in docs]


if __name__ == "__main__":
    # Configure the Streamlit page
    st.set_page_config(
        layout="wide",
        page_icon="./image/logo.svg",
        page_title="RAGXplorer!",
    )
    st.title("RAGXplorer")

    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader(
            "Upload your file (e.g., text, pdf)",
            type=["txt", "pdf"],
            accept_multiple_files=True, 
            help="Choose one or more files for retrieval."
        )
        if uploaded_file:
            for file in uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file.getbuffer())
                    tmp_path = tmp_file.name

                if file.type == 'pdf':
                    loaders.append(PyPDFLoader(tmp_path))

        for loader in loaders:
            documents.extend(split_document(loader))
            vectorstore = create_vs(documents)

    st.write("Welcome to **RAGXplorer**!")
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    for index, (q, exp) in enumerate(st.session_state.conversation):
        col1, col2 = st.columns([3, 1.2])
        with col1:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {exp['Answer']}")
            st.markdown(
                f"""<p style="color:gray; font-style:italic;">Exp: {exp['Explanation']}</p>""",
                unsafe_allow_html=True,)

        with col2:
            with st.expander("Details", expanded=True):
                st.markdown(f"**Relevancy**: {exp['Relevancy']}")
                st.markdown(f"**Faithfulness**: {exp['Faithfulness']}")
                st.markdown(f"**Context Quality**: {exp['Context Quality']}")
                st.markdown(f"**Needs Web Search**: {exp['Needs Web Search']}")

    col1, col2 = st.columns([4, 1])  
    with col1:
        question = st.text_input(
            "Ask a question:",
            placeholder="Type your question here...",
        )

    with col2:
        st.write("")
        st.write("")
        if st.button("Ask", key="ask_button", type="primary"):
            if question.strip():
                response = groq_agent.run_sync(question)
                st.session_state.conversation.append((question, json.loads(response.data)))
                st.rerun()
            else:
                st.warning("Please enter a question before clicking 'Ask'.")