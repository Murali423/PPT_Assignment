from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#Extracting the data
def load_data(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


#create chunks for document
def text_split(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 100, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

