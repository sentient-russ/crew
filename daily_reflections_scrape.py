#pip install ollama langchain beautifulsoup4 chromadb gradio
import bs4 #beautiful soup
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# pip install selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

chrome_drv_path = "/crew/chromedriver-linux64/chromedriver"
driver_service = Service(executable_path=chrome_drv_path)
driver = webdriver.Chrome(service=driver_service)
driver.get("https://www.aa.org/")
driver.find_element_by_class_name('read-more').click()


loader = WebBaseLoader(
    verify_ssl=False,
    web_paths=("https://www.aa.org/",),
    
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            ['article',],
        )
    ),
)
docs = loader.load()
replacement_str1 = docs[0].page_content.replace("\n","")
replacement_str2 = replacement_str1.replace("\r","")
replacement_str3 = replacement_str2.replace("  ","")
docs[0].page_content = replacement_str3

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #init splitter
split_text = text_splitter.split_documents(docs)

# create vector store
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=split_text, embedding=embeddings)
# init retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) #search_kwargs={"k": 1} allows retriever to accept there is only one document in the vectore store.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) #merges all retrieved documents before sending to LLM so they can be used as a context

def ask_ai(question, context):
    llm = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_template("""Your are a sophisticated funny helpful anonymous cybertronic AA member and writer that does not mention yourself or compliment the reader. Write only on the provided context and AA literature:
    <context>
    {context}
    </context>
    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt) #prepares the llm to use the prompt
    retrieval_chain = create_retrieval_chain(retriever, document_chain) #prepares llm <-- promt respone to be added to non-persistent vector store
    return retrieval_chain.invoke({"input": question},{"context": context})

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ask_ai(question, formatted_context)

result1 = rag_chain("Based on the content of the big book of alcoholics anonymous give the main topic contained in the context.")
print("")

result2 = rag_chain("Write about the topic TOPIC:/n folled by a numbered list RELATED STEPS:/n of three steps from the book 12 steps and 12 traditions of Alcoholics Anonymous. Include with each step how they are relate to the following context" + result1["answer"])
print(result2["answer"])
#result3 = rag_chain("Please show us your cleaver consise nature by adding details to the follwing list that are relavant to recovery from alcoholism: " + result2["answer"])
#print(result3["answer"])