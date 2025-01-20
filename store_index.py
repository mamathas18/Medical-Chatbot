from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the index name and parameters
index_name = "medicalbot"
dimension = 384  # Match the dimension of your embedding model
metric = "cosine"

try:
    # List all existing indexes
    existing_indexes = pc.list_indexes()
    print(f"Existing indexes: {existing_indexes}")

    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists. Skipping creation.")
    else:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' created successfully!")
except PineconeApiException as e:
    if "ALREADY_EXISTS" in str(e):
        print(f"Index '{index_name}' already exists. No need to recreate.")
    else:
        print(f"Error creating index: {e}")
        raise

# Load and preprocess data
print("Loading PDF data...")
extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)

# Load embedding model
print("Downloading embeddings model...")
embeddings = download_hugging_face_embeddings()

# Embed each chunk and upsert into Pinecone
try:
    print(f"Upserting documents into the '{index_name}' index...")
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings,
    )
    print("Documents upserted successfully!")
except Exception as e:
    print(f"Error during document upsert: {e}")
    raise
