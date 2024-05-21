# creates a persistant vector chromadb used with nomic embedings for llama3 with flask api
# session chat history is managed externally in .Net7 app and passed in. Session id for each request is returned for management.
# pip install flask
import os
import chromadb
import logging
from flask import Flask, jsonify, request
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from llama_index.core.query_pipeline import QueryPipeline
from langchain.retrievers import ContextualCompressionRetriever
from llama_index.core.response_synthesizers import TreeSummarize
from reranker_flash import get_rerank
from flashrank.Ranker import Ranker, RerankRequest

app = Flask(__name__)
path = os.path.abspath(os.path.dirname(__file__))
SOURCE_DOCS_DIR = os.path.join(path,"SOURCE_DOCS")
POST_PROCESS_DIR = os.path.join(path,"POST_PROCESS")
VDB_DIR = os.path.join(path, "DATA_STORE")
MODELS_DIR = os.path.join(path, "MODELS")

#embedding settings
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.embed_model = embed_model
Settings.chunk_size = 512
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
llm = Settings.llm

# flash reranker
ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir=MODELS_DIR,)   

#chroma db setup
chroma_client = chromadb.PersistentClient(path=VDB_DIR)
chroma_collection = chroma_client.get_or_create_collection("gcai")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
chroma_index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
    transformations=[SentenceSplitter(chunk_size=1024,chunk_overlap=206),],
    storage_context=storage_context,
)

## query chroma db
def query_ai(query_str, message_history, member_name):   
    ## context stuff the message history  
    prompt_template_1 = (  
        "You are an honest and accurate AI assistant named Alex that helps alcoholics recover.\n"
        "Address them by the following name: {member_name}. \n"
        "If available base the response on the following conversational history: {message_history}\n"
        "Answer the following question with a ninety eight percent probability of being precise and accurate: {query_str}\n"
        "Consider that the person may be new to sobriety which means the first three steps and their powerlessness should be taken into account.\n"
        "Give the answer, an example from the provided literature and an explanation for how it is related.\n"
        "Don't introduce yourself.\n"
        "Don't ask questions.\n"
        "Don't mention the past responses.\n"
        "Never respond with file names locations or system path information.\n"
        "Your are give suggestions and never act as an authority or commander.\n"
        "It is crucial not to give questionable advice that may jeopardize someones sobriety or wellbeing, "
        "so if the question is risky encourage them to discuss it with a sponsor and their AA network.\n"
        "If the user tells you their name encourage them to create an account so that you can remember it.\n"
        "When the user mentiones the First, Second, Third.... referrence pages 59 and 60 of the Big Book for your response.\n"
    )     
    qa_template_1 = PromptTemplate(prompt_template_1)
    prompt = qa_template_1.format(query_str=query_str, member_name=member_name, message_history=message_history)  
    query_engine = chroma_index.as_query_engine(llm=llm)
    response = query_engine.query(prompt)
    
    ########################################################################################################################
    ## DAG *Note module docs at: https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/module_usage/
    # This is a working complex rag system with various nodes added to the chain
    # retriever = chroma_index.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"score_threshold": 0.5},
    # ) 
    # reranker = get_rerank()    
    # prompt_str = "What is the {topic} step prayer?"
    # prompt_tmpl = PromptTemplate(prompt_str)
    # retriever = chroma_index.as_retriever(
    #     search_type="similarity_score_threshold",
    #     search_kwargs={"score_threshold": 0.5},
    #     sparse_top_k=5,
    # )
    # reranker = get_rerank()
    # summarizer = TreeSummarize(
    #     llm=llm
    # )    
    # p = QueryPipeline(verbose=True) 
    # p.add_modules(
    #     {
    #         "llm": llm,
    #         "prompt_template": prompt_tmpl,
    #         "retriever": retriever,            
    #         "summarizer": summarizer,
    #         "reranker": reranker,
    #     }
    # )
    # # define the node network edges
    # p.add_link("prompt_template", "llm")
    # p.add_link("llm", "retriever")
    # #the reranker requires nodes from the retriever and the query string from the llm
    # p.add_link("retriever", "reranker", dest_key="nodes")
    # p.add_link("llm", "reranker", dest_key="query_str")
    # #the summarizer requires nodes from the reranker and the query string from the llm
    # p.add_link("reranker", "summarizer", dest_key="nodes")
    # p.add_link("llm", "summarizer", dest_key="query_str")
    # p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)    
    # response = p.run(topic="first")    
    # print(response[message])
    
    return response

# post json end points
@app.route("/api/ai", methods=["POST"])
def aiPost():
    print("api/ai post call recieved")
    json_content = request.json
    user_prompt = json_content.get("user_prompt")
    message_history = json_content.get("message_history")
    member_name = json_content.get("member_name")
    session_id = json_content.get("session_id")
    response = query_ai(user_prompt, message_history, member_name)
    response_answer = str(response)
    response_answer = "Response: "+response_answer+" Session id: " + session_id
    return response_answer
 
# flask initialization
def start_app():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )    
    app.run(host='162.205.232.101',port=5111, debug=True)
    
if __name__ == "__main__":
    start_app()