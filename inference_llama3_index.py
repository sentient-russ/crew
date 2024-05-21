# uses persistant vector index used with nomic embedings for llama3
# an index must be created with the ingest_llama3_index.py prior to use
import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

path = os.path.abspath(os.path.dirname(__file__))
SOURCE_DOCS_DIR = os.path.join(path,"SOURCE_DOCS")
POST_PROCESS_DIR = os.path.join(path,"POST_PROCESS")
VDB_DIR = os.path.join(path, "DATA_STORE")
MODELS_DIR = os.path.join(path, "MODELS")

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

storage_context = StorageContext.from_defaults(persist_dir=VDB_DIR)
index = load_index_from_storage(storage_context)

# and the query
query_engine = index.as_query_engine()
response = query_engine.query("What is the first step prayer?")
print(response)

