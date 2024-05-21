<p align="center"><img width="475" alt="demo-screen-weather" src="https://github.com/sentient-russ/crew/assets/108576049/74233a76-cded-4071-bfba-67c75997327f"></p>

### Overview:
This repo contains a collection of AI assistants and tooling. Being on the forefront of the AI craze means they are expected to be replaced with better options when they become available. 🤖🚀

### **High-level technologies:**
llama3, langchain, BeautifulSoup, python

### **Utilities:**
> - scrape_daily_reflections.py, Collects the days reflection and then iterates it through AI LLM to generate an output summary of the reflection
> - scrape_webpage.py, Scrapes information from MagnaDigi.com's about page and summarizes it with AI LLM
> - ingest_langchain_universal.py, Loads and stores source documents into a chroma vector store for huggingface langchain inference
> - ingest_llama3_index.py, Scaffolds directories, loads data, and stores loaded documents in persistent llama index store for inference
> - ingest_llama3_chroma.py, Scaffolds directories, loads data, and stores loaded documents in a persistent chroma vector store
> - inference_llama3_chroma_api.py, Flask API that processes natural language requests using llama3, DAG piplines and chroma to generate responses (Will be adding MySQL into the pipeline soon)
> - inference_llama3_index.py, Sample for inference on llama.index store and LLM inference using Llama3
> - reranker_flash.py, Reranking processor for handling lost data in the middle of retrieval results for use in DAG pipelines



