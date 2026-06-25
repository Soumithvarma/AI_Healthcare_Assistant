from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Load TXT files
# -----------------------------
loader = DirectoryLoader(
    "../medical_knowledge",
    glob="**/*.txt",
    loader_cls=TextLoader
)

documents = loader.load()

print(f"Loaded {len(documents)} documents")

# -----------------------------
# Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks")

# -----------------------------
# Embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create ChromaDB
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="../chroma_db"
)

vectorstore.persist()

print("ChromaDB created successfully!")