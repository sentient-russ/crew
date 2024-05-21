# creates a persistant vector index used with nomic embedings for llama3
# scafolds SOURCE_DOCUMENTS, POST_PROCESS, DATA_INDEX directories
# the txt of pdf documents that will be indexed must be coppied to a directory named
# SOURCE_DOCUMENTS in the same location as this python file
import os
import shutil
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

path = os.path.abspath(os.path.dirname(__file__))
SOURCE_DOCS_DIR = os.path.join(path,"SOURCE_DOCS")
POST_PROCESS_DIR = os.path.join(path,"POST_PROCESS")
VDB_DIR = os.path.join(path, "DATA_STORE")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.chunk_size = 512
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
storage_context = StorageContext.from_defaults(persist_dir=VDB_DIR)
   
if len(os.listdir(SOURCE_DOCS_DIR)) == 0:
    print("./SOURCE_DOCUMENTS directory is empty. No new indexing will be conducted.")
else:    
    print("New files found in ./SOURCE_DOCUMENTS Indexing...")
    if not os.path.exists(VDB_DIR):
        os.mkdir(DATA_INDEX_DIR)        
        documents = SimpleDirectoryReader(SOURCE_DOCS_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=512)])
        index.storage_context.persist(persist_dir=VDB_DIR)
        if not os.path.exists(POST_PROCESS_DIR):
            os.mkdir(POST_PROCESS_DIR) 
        file_names = os.listdir(SOURCE_DOCS_DIR)
        for file_name in file_names:
            shutil.move(os.path.join(SOURCE_DOCS_DIR, file_name), POST_PROCESS_DIR)
        else:
            file_names = os.listdir(SOURCE_DOCS_DIR)
            for file_name in file_names:
                shutil.move(os.path.join(SOURCE_DOCS_DIR, file_name), os.path.join(POST_PROCESS_DIR, file_name))      
        print("Indexing Complete...")        
    else:
        documents = SimpleDirectoryReader(SOURCE_DOCS_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents,transformations=[SentenceSplitter(chunk_size=512)])
        index.storage_context.persist(persist_dir=VDB_DIR)
        if not os.path.exists(POST_PROCESS_DIR):
            os.mkdir(POST_PROCESS_DIR) 
        file_names = os.listdir(SOURCE_DOCS_DIR)
        for file_name in file_names:
            shutil.move(os.path.join(SOURCE_DOCS_DIR, file_name), POST_PROCESS_DIR)
        else:
            file_names = os.listdir(SOURCE_DOCS_DIR)
            for file_name in file_names:
                shutil.move(os.path.join(SOURCE_DOCS_DIR, file_name), os.path.join(POST_PROCESS_DIR, file_name))      
        print("Indexing Complete...")    

index = load_index_from_storage(storage_context)
# query the index store
query_engine = index.as_query_engine()
response = query_engine.query("What is the first step prayer?")
print(response)