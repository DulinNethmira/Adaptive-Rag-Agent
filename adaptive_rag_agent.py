from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

class RAGState(TypedDict):
    question: str
    context: str
    is_relevant: bool
    answer: str
    enriched_question: str
    web_results: str

loader = PyPDFLoader('hr_manual.pdf')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

text_splits = text_splitter.split_documents(loader.load())

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = InMemoryVectorStore(embedding=embedding_model)

vector_store.add_documents(documents=text_splits)

model = ChatOpenAI(model="gpt-5.2-2025-12-11")

def retrieve_context(enhanced_user_query: str):
    """Retrieve information to help answer user queries using enhanced_query returned from the enhance_user_query tool"""
    retrieved_documents = vector_store.similarity_search(enhanced_user_query, k=5)

    content = "\n\n".join(
        f"Source : {doc.metadata} \n Content : {doc.page_content}" for doc in retrieved_documents
    )
    return content

def validate_context(question: str, context: str):
    validation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a strict validator.
        Return ONLY 'YES' or 'NO'.
        Answer YES if the context contain information that can directly answer the question
        """),
        ("human", """
        Question : {question}
        
        Context : {context}
        """)
    ])

    validation_chain = validation_prompt | model
    result = validation_chain.invoke({
        "question" : question,
        "context" : context
    })

    return result.content.strip() == "YES"  # Returns a True or False boolean.

def retrieve_node(state: RAGState):
    context = retrieve_context(state["enriched_question"])
    return {"context" : context}

def validate_node(state: RAGState):
    is_relevant = validate_context(state["question"],
                                   state["context"])

    return {"is_relevant" : is_relevant}

def answer_node(state: RAGState):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistance that answers question based on the provided context
                      User Only the information from the context to answer"""),
        ("human", """ Context: {context}
                     Question: {question} """)
    ])

    rag_chain = prompt_template | model
    response = rag_chain.invoke({
        "context" : state["context"],
        "question" : state["question"]
    })
    return {"answer" : response.content}

def enrich_query_node(state: RAGState):
    enrich_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You rewrite user queries to improve document retrieval.
        Make the question more specific and focused.
        Do not answer the question.
        """),
        ("human", """
        Original Question : {question}
        
        Rewrite the question for better search
        """)
    ])

    enrich_chain = enrich_prompt | model

    enriched_results = enrich_chain.invoke({
        "question" : state["question"]
    })

    return {"enriched_question" : enriched_results.content}

def web_search_node(state: RAGState):
    web_search = web_search_tool.invoke({
        "query" : state["question"]
    })

    results = web_search["results"]

    web_results_joined = "\n".join(
        [result["content"] for result in results]
    )

    return {"web_result" : web_results_joined}

def web_answer_node(state: RAGState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You answer questions using web search results.
        If the answer is uncertain, say so.
        """),
        ("human", """
        question : {question}
        
        Web Search Results:
        {web_result}
        """)
    ])

    chain = prompt | model

    response = chain.invoke({
        "question" : state["question"],
        "web_result" : state["web_results"]
    })