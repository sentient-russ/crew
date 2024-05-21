# pip install -q llama-index
# pip install -q tranformers
# pip install -q accelerate
# pip install -q optimum[exports]
# pip install -q InstructorEmbedding
# pip install -q pypdf
# pip install -q llama-index chromadb
# pip install -q chromadb
# pip install -q sentence_transformers
# pip install -q pydantic==1.10.11
# pip install llama-index-vector-stores-chroma

# creates a persistant vector db used with nomic embedings for llama3
# scafolds SOURCE_DOCUMENTS, POST_PROCESS, DATA_INDEX directories
# the txt and pdf documents that will be indexed must be coppied to a directory named
# SOURCE_DOCUMENTS in the same location as this python file
import os
import shutil
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

path = os.path.abspath(os.path.dirname(__file__))
SOURCE_DOCS_DIR = os.path.join(path,"SOURCE_DOCS")
POST_PROCESS_DIR = os.path.join(path,"POST_PROCESS")
VDB_DIR = os.path.join(path, "DATA_STORE")

#chroma db setup
chroma_client = chromadb.PersistentClient(path=VDB_DIR)
chroma_collection = chroma_client.get_or_create_collection(name="gcai")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.chunk_size = 512
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# scafolds ingestion directories, embeds, and backs up source documents
if len(os.listdir(SOURCE_DOCS_DIR)) == 0:
    print("./SOURCE_DOCUMENTS directory is empty. No new indexing will be conducted.")
else:    
    print("New files found in ./SOURCE_DOCUMENTS Indexing...")
    if not os.path.exists(VDB_DIR):
        os.mkdir(VDB_DIR) 
        documents = SimpleDirectoryReader(SOURCE_DOCS_DIR).load_data()
        gc_chroma = VectorStoreIndex.from_documents(documents, storage_context=storage_context,transformations=[SentenceSplitter(chunk_size=512)])
        gc_chroma.storage_context.persist(persist_dir=VDB_DIR)
        if not os.path.exists(POST_PROCESS_DIR):
            os.mkdir(POST_PROCESS_DIR) 
            file_names = os.listdir(SOURCE_DOCS_DIR)
            for file_name in file_names:
                shutil.move(os.path.join(SOURCE_DOCS_DIR, file_name), os.path.join(POST_PROCESS_DIR, file_name))   
        else:
            file_names = os.listdir(SOURCE_DOCS_DIR)
            for file_name in file_names:
                shutil.move(os.path.join(SOURCE_DOCS_DIR, file_name), os.path.join(POST_PROCESS_DIR, file_name))      
        print("Indexing Complete...")    
    else:
        documents = SimpleDirectoryReader(SOURCE_DOCS_DIR).load_data()
        gc_chroma = VectorStoreIndex.from_documents(documents, storage_context=storage_context,transformations=[SentenceSplitter(chunk_size=512)])
        gc_chroma.storage_context.persist(persist_dir=VDB_DIR)
        if not os.path.exists(POST_PROCESS_DIR):
            os.mkdir(POST_PROCESS_DIR) 
            file_names = os.listdir(SOURCE_DOCS_DIR)
            for file_name in file_names:
                shutil.move(os.path.join(SOURCE_DOCS_DIR, file_name), os.path.join(POST_PROCESS_DIR, file_name))   
        else:
            file_names = os.listdir(SOURCE_DOCS_DIR)
            for file_name in file_names:
                shutil.move(os.path.join(SOURCE_DOCS_DIR, file_name), os.path.join(POST_PROCESS_DIR, file_name))     
        print("Indexing Complete...")  

# Load store (Remove after testing)
chroma_client = chromadb.PersistentClient(path=VDB_DIR)
chroma_collection = chroma_client.get_or_create_collection("gcai")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
chroma_index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
    show_progress=True,
)

# in any event query the index
query_engine = chroma_index.as_query_engine()
response = query_engine.query("What is the first step prayer?")
print(response)

query_engine = chroma_index.as_query_engine()
response = query_engine.query("What does the the Big Book say about the tenth step?")
print(response)
