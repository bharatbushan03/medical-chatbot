from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
from src.prompt import *
import os
import uvicorn

app = FastAPI(title="Medical Chatbot")

templates = Jinja2Templates(directory="templates")

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HuggingFace_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY or ""
os.environ['HUGGINGFACE_API_TOKEN'] = HuggingFace_API_TOKEN or ""

embeddings = download_embeddings()

index_name = 'medical-chatbot'
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={'k': 3})

# --- LLM Setup ---
# Use AutoModelForSeq2SeqLM + task="text-generation" to bypass environment-specific issues.
model_id = 'google/flan-t5-small'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
# NOTE: 'text-generation' is the only valid text task in this registry despite it being a Seq2Seq model.
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.3
)
llm = HuggingFacePipeline(pipeline=pipe)

# --- Prompts with chat history ---
# Reformulation prompt: Converts conversational question to standalone question
contextualize_q_system_prompt = (
    "Given the chat history and the latest user question, "
    "reformulate the question so it can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed, otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Final QA prompt
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answering_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answering_chain)

# --- Session Store for Memory ---
session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/chat')
async def chat(chat_request: ChatRequest):
    question = chat_request.question.strip()
    if not question:
        return {'error': 'question is required'}

    response = conversational_rag_chain.invoke(
        {'input': question},
        config={'configurable': {'session_id': chat_request.session_id}},
    )
    answer = response.get('answer', '')
    return {'answer': answer}


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8080, reload=False)