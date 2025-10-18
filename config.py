"""
Project configuration.

How to use:
- Create a file named `.env` in the project root (same folder as this file).
- Copy the contents from `.env.example` (we'll generate one) and fill in real values.
- These settings are loaded via python-dotenv at import time.

Never commit your real keys to git.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a `.env` file if present
load_dotenv()

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Pinecone (v3)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
# Serverless: cloud provider and region, used when creating the index
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "gcp")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vietnam-travel")
PINECONE_VECTOR_DIM = int(os.getenv("PINECONE_VECTOR_DIM", "1536"))
