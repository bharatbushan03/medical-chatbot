from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, download_embeddings, text_split
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
HuggingFace_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['HUGGINGFACE_API_TOKEN'] = HuggingFace_API_TOKEN

extracted = load_pdf_files(r'data')
filter_data = filter_to_minimal_docs(extracted)
text_chunks = text_split(filter_data)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key = pinecone_api_key)

index_name = 'medical-chatbot'

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension = 384,
        metric = 'cosine',
        spec = ServerlessSpec(cloud = 'aws', region = 'us-east-1')
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    index_name = index_name
)

