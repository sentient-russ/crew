#pip install ollama langchain beautifulsoup4 chromadb gradio

import bs4 #beautiful soup
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

loader = WebBaseLoader(
    verify_ssl=False,
    web_paths=("https://magnadigi.com/Home/About?area=About",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer('p',
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
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
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

result = rag_chain("Please give a summery of the provided context.")
print(result["answer"])