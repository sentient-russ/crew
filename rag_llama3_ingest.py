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

path = "/home/russrex/crew/repo/"
DATA_INDEX_DIR = os.path.join(path,"DATA_INDEX")
SOURCE_DOCUMENTS_DIR = os.path.join(path,"SOURCE_DOCUMENTS")
POST_PROCESS_DIR = os.path.join(path,"POST_PROCESS")

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

if len(os.listdir(SOURCE_DOCUMENTS_DIR)) == 0:
    print("./data directory is empty. No new indexing will be conducted.")
else:    
    print("New files found in ./data. Indexing...")
    if not os.path.exists(DATA_INDEX_DIR):
        os.mkdir(DATA_INDEX_DIR) 
    else:
        documents = SimpleDirectoryReader(SOURCE_DOCUMENTS_DIR).load_data() #This works with text and PDF's
        index = VectorStoreIndex.from_documents(
            documents,
        )
        # save to persistant index storage
        index.storage_context.persist(persist_dir=DATA_INDEX_DIR)
        # create post process dir if it does not exist
        if not os.path.exists(POST_PROCESS_DIR):
            os.mkdir(POST_PROCESS_DIR) 
        # move indexed files to post process directory
        file_names = os.listdir(SOURCE_DOCUMENTS_DIR)
        for file_name in file_names:
            shutil.move(os.path.join(SOURCE_DOCUMENTS_DIR, file_name), POST_PROCESS_DIR)
       
    
# check if storage already exists
if not os.path.exists(DATA_INDEX_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader(SOURCE_DOCUMENTS_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=DATA_INDEX_DIR)
else:
    # create post process dir if it does not exist
    if not os.path.exists(DATA_INDEX_DIR):
        os.mkdir(DATA_INDEX_DIR) 
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=DATA_INDEX_DIR)
    index = load_index_from_storage(storage_context)

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)