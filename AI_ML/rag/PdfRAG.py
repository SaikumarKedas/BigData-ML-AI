from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# Start Local LLM
llm = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")

# Load PDF
loader = PyPDFLoader("AI_ML/data/cricket.pdf")
pages = loader.load()
print("\n Total Pages:", len(pages))

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=60)
chunks = text_splitter.split_documents(pages)
print("\n Total Chunks:", len(chunks))

# Embedding
embedding = OllamaEmbeddings(model="nomic-embed-text")

# Create Vector Database
db = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="AI_ML/VectorDB/")
retriever = db.as_retriever(search_kwargs={"k": 3})
print("\n Vector Database Created and Persisted at: AI_ML/VectorDB/")


# LLM for processing the retrieved documents
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."),
    ("system", "{history}"),
    ("system", "{context}"),
    ("human",  "{input}")
])

chat_history = []
print("\n PDF RAG Created, You can now ask questions based on the PDF content. Type 'exit' to quit.")


# Conversation loop
while True:
    user_input = input("\nUser: ")
    if user_input.lower() == "exit":
        print("Exiting the conversation. Goodbye!")
        break

    # Retrieve relevant documents based on user input
    retrieved_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Generate response using LLM
    response = llm.invoke(
        prompt.format(history="\n".join(chat_history), context=context, input=user_input)
    )

    # Print the response
    print("\nAssistant:", response.content)

    # Update chat history
    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Assistant: {response}")










