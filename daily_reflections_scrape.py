# pip install ollama langchain beautifulsoup4 chromadb gradio
# pip install jq
import bs4 
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
from langchain_core.documents import Document
from langchain_community.document_loaders import JSONLoader
import urllib.request, json 
import datetime

current_time = datetime.datetime.now()
current_day_int = current_time.day
current_month_int = current_time.month
month_day = {"month":"","day":""}
if current_day_int < 10:
    month_day["month"] = "0" + str(current_day_int)
else:
    month_day["month"] = str(current_day_int)

if current_month_int < 10:
    month_day["day"] = "0" + str(current_month_int)
else:
    month_day["day"] = str(current_month_int)    
api_string = "https://www.aa.org/api/reflections/" + month_day['day'] + "/" + month_day['month'] + "/"

with urllib.request.urlopen(api_string) as url:
    json_data = json.load(url)
itr_string1 = json_data['data']
itr_string2 = itr_string1.replace("\n", "")
itr_string3 = itr_string2.replace('\t', "")
itr_string4 = itr_string3.replace('  ', "")
final_string = itr_string4.replace('\xa0', "")

json_array = []
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_data(self, data):
        json_array.append(data)
parser = MyHTMLParser()
parser.feed(final_string)

print("")
ref_title = json_array[0]
ref_date = json_array[1]
ref_passage = json_array[2]
ref_passage_source = json_array[3]
ref_experience_p1 = json_array[6]
ref_experience_p2 = json_array[7]
ref_experience_p3 = json_array[8]
ref_experience_p4 = json_array[9]

context_string = "Title: " + ref_title + ", Date: " + ref_date + ", Passage: " + ref_passage + ", Passage Source: " + ref_passage_source + ", Member Experience: " + ref_experience_p1 + " " + ref_experience_p2 + " " + ref_experience_p3 + " " + ref_experience_p4

docs = [Document(page_content=context_string)] 

# Chunk Data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #init splitter
split_text = text_splitter.split_documents(docs)

# Create Vector Store
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=split_text, embedding=embeddings)

# Init Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) #search_kwargs={"k": 1} allows retriever to accept there is only one document in the vectore store.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs) #merges all retrieved documents before sending to LLM so they can be used as a context

# Query llama3
def ask_ai(question, context):
    llm = Ollama(model="llama3")
    prompt = ChatPromptTemplate.from_template("""Your are a sophisticated funny helpful anonymous cybertronic AA member and writer that does not reveal anything about yourself or compliment the reader. Write only on the provided context and AA literature:
    <context>
    {context}
    </context>
    Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt) #prepares the llm to use the prompt
    retrieval_chain = create_retrieval_chain(retriever, document_chain) #prepares llm <-- promt respone to be added to non-persistent vector store
    return retrieval_chain.invoke({"input": question},{"context": context})

# Build Chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question) #retreives embeded chunks from daily reflection store which are relevant to the question
    formatted_context = format_docs(retrieved_docs) #merges chunks
    return ask_ai(question, formatted_context)
print("-------------------------------------------------------")
result1 = rag_chain("Please return the main topic and date contained in the context under the headings TITLE: and DATE: , do not use asterisks")
print(result1["answer"])
print("AI generated based on Daily Reflections found at: https://www.aa.org/daily-reflections")
print("")
result2 = rag_chain("Write an insightful summery about the following passage: " + context_string + ", with the heading PASSAGE SUMMERY: followed by one to three insightful paragraphs that relate the passage and experience to the principles in the AA 12 steps with the heading of STEP RELATED DISCUSSION: , without quoting the litterature.")
print(result2["answer"])

result3 = rag_chain("Respond like a loving sober drunk would in a short paragraph about the passage under the heading of THE WILD RUMPUS COMENTARY: ")
print("")
print(result3["answer"])
print("-------------------------------------------------------")
