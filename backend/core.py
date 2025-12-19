from dotenv import load_dotenv

load_dotenv()
from typing import Any, Dict, List

from langchain import hub
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from consts import INDEX_NAME

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    """
    Run RAG chain with source documents using the latest LangChain retrieval chain.
    Returns both the answer and source documents.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    # result contains: {"input": ..., "chat_history": ..., "context": [source_docs], "answer": ...}
    return result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def run_llm_with_sources(query: str, chat_history: List[Dict[str, Any]] = []):
    """
    Modern LangChain LCEL approach for retrieval with source documents.
    Uses RunnableParallel to return both the answer and source documents.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    chat = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever = docsearch.as_retriever()

    # Chain to get the answer
    answer_chain = (
        retrieval_qa_chat_prompt
        | chat
        | StrOutputParser()
    )

    # Use RunnableParallel to retrieve docs and pass through query
    # Then use assign to add the answer while keeping source documents
    chain = (
        RunnableParallel(
            context=retriever,
            input=RunnablePassthrough()
        )
        .assign(
            answer=lambda x: answer_chain.invoke({
                "context": format_docs(x["context"]),
                "input": x["input"]
            })
        )
    )

    result = chain.invoke(query)
    # result contains: {"context": [source_docs], "input": query, "answer": ...}

    # Format the response with source documents
    return {
        "answer": result["answer"],
        "source_documents": result["context"],
        "sources": list(set([
            doc.metadata.get("source", "Unknown")
            for doc in result["context"]
        ]))
    }
