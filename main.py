import os

from langchain.prompts import PromptTemplate
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ChatMessage
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.globals import set_debug

from config import MODEL_PATH, CHROMADB_DIR, NO_RAG, CONTEXTUALIZE_PROMPT, QA_PROMPT, VERBOSE

set_debug(VERBOSE)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

contextualize_system_prompt = CONTEXTUALIZE_PROMPT
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

system_prompt = QA_PROMPT

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

custom_rag_prompt = PromptTemplate.from_template(system_prompt)

print("Загружаю модель...")
llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=2048,
    verbose=False
)
print("Модель загружена!")

if NO_RAG:
    vectorstore = Chroma()
else:
    if os.path.exists("./chromadb"):
        print("Загружаю векторизацию ГОСТ...")
        vectorstore = Chroma(
            persist_directory=CHROMADB_DIR,
            embedding_function=LlamaCppEmbeddings(model_path=MODEL_PATH, verbose=False)
        )
        print("ГОСТ загружен!")
    else:
        print("Загружаю ГОСТ...")
        loader = UnstructuredMarkdownLoader("gost.md")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(data)
        print("ГОСТ загружен!")
        print("Векторизую ГОСТ...")
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=LlamaCppEmbeddings(model_path=MODEL_PATH, verbose=False),
            persist_directory=CHROMADB_DIR
        )
        print("ГОСТ векторизован!")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

contextualize_chain = contextualize_prompt | llm | StrOutputParser()

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_chain
    else:
        return input["question"]
    
rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)

chat_history = []
print()
print("Привет! Я знаю все о ГОСТ 27003. Спроси меня о чем-нибудь!")

while True:
    question = input("Вопрос: ")
    print("Думаю...")
    context = {"question": question, "chat_history": chat_history}
    ai_msg = rag_chain.invoke(context)
    chat_history.extend([
        ChatMessage(role="Человек", content=question), 
        ChatMessage(role="Ассистент", content=ai_msg),
    ])
    chat_history = chat_history[-4:]
    print(f"Ответ: {ai_msg}")
    print()