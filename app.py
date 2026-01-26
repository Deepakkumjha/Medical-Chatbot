import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    if not os.path.exists(DB_FAISS_PATH):
        st.error("FAISS vectorstore not found")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_prompt():
    return PromptTemplate(
        template="""
Use the information in the context to answer the question.
If you do not know the answer, say you do not know.
Do not add anything outside the context.

Context:
{context}

Question:
{question}

Answer directly.
""",
        input_variables=["context", "question"],
    )


def main():
    st.set_page_config(page_title="Medical Chatbot")
    st.title("Medical Chatbot")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not set")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_input = st.chat_input("Ask your question")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        vectorstore = get_vectorstore()

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                groq_api_key=groq_api_key,
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": get_prompt()},
        )

        response = qa_chain.invoke({"query": user_input})

        answer = response["result"]
        sources = response["source_documents"]

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.chat_message("assistant").markdown(
            "Source Docs:\n\n" + str(sources)
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": str(sources)}
        )


if __name__ == "__main__":
    main()
