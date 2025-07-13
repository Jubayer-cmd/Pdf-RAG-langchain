import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Initialize the LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=SecretStr(gemini_key))
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(gemini_key))

# Load PDF file
pdf_path = "./data/pdfs/Project_proposal.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

# Create or update the vector store
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory="./data/vectorstore",
    collection_name="pdf_rag_collection"
)


# Set up the retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create a prompt template for the QA chain
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

#chain using LCEL
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def main():
    print("PDF processed and vector store created/updated.")
    
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        if query:
            try:
                result = qa_chain.invoke(query)
                print("\nAnswer:")
                print(result)
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
