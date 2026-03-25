from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

# Loading the PDF files from the specified directory
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Filtering the minimal docs
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    '''
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    '''
    minimal_docs: List[Document] = []
    for doc in docs:
        # Pinecone rejects None metadata values; keep only a safe string
        src = doc.metadata.get('source') or doc.metadata.get('soruce') or ""
        minimal_docs.append(Document(page_content=doc.page_content, metadata={'source': str(src)}))

    return minimal_docs

def download_embeddings():
    '''
    Download and return the HuggingFace Embeddings model.
    '''

    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

def text_split(minimal_data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    texts_chunks = splitter.split_documents(minimal_data)
    return texts_chunks

