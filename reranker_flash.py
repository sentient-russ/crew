# flashrank's smallest model is < 4MB which makes it ideal for cloud data usage constraints/costs. It also mitigates api data transfer privacy concerns
# Liscense MIT
# Default port 8501
from flashrank.Ranker import Ranker, RerankRequest
import os

path = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(path, "MODELS")

# ranker = Ranker()
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=MODELS_DIR)
#ranker = Ranker(model_name="rank-T5-flan", cache_dir=MODELS_DIR)
#ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir=MODELS_DIR)

def get_rerank(query_str="", nodes=[]):

    if query_str=="" or nodes==[]:
        print("Flash rank missing arguments.")
        return []
    else:
        rerankrequest = RerankRequest(query=query_str, passages=nodes)
        results = ranker.rerank(rerankrequest)
        print(results)
        return results

print("Flash Rank Import successful!")

 