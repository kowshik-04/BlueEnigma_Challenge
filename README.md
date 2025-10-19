# 🧠 Hybrid Knowledge AI System  
### Blue Enigma Labs – AI Engineer Technical Challenge  
**Author:** Mente Rama Naga Kowshik  

## 🚀 Overview  

This project implements a **Hybrid Retrieval + Reasoning AI System** that integrates:  
- 🧩 **Pinecone** – for semantic vector retrieval  
- 🌐 **Neo4j** – for graph-based relational context  
- 🤖 **OpenAI GPT Models** – for natural language reasoning and response generation  

The system intelligently combines **meaning (semantic embeddings)** and **structure (graph relationships)** to generate rich, contextual, and human-like answers.  

> Example Query:  
> _“Create a romantic 4-day itinerary for Vietnam.”_

The assistant retrieves semantically similar locations from Pinecone, enriches them with connected nodes from Neo4j, and uses OpenAI’s reasoning to produce a complete, context-aware travel itinerary.

---

## 🧩 Architecture  

User Query -> OpenAI Embedding → Vector Representation -> Pinecone (Semantic Search) -> Top-k Semantic Matches -> Neo4j (Graph Reasoning) -> Combined Context (Vector + Graph) -> OpenAI Chat Model (Reasoning & Generation) -> Final Context-Aware Answer


## 🧠 Core Components  

### 1️⃣ Pinecone – Semantic Search  
- Uses `text-embedding-3-small` for embeddings  
- Stores and retrieves vectorized travel data  
- Supports metadata and contextual similarity queries  

### 2️⃣ Neo4j – Graph Reasoning  
- Captures structured relationships (city → attraction → cuisine)  
- Enables context enrichment and relational traversal  

### 3️⃣ OpenAI GPT Models  
- Generates embeddings and natural language responses  
- Uses `gpt-4o-mini` for reasoning and synthesis  

---

## 🧰 Tech Stack  

| Component | Technology |
|------------|-------------|
| Language | Python 3.10+ |
| Vector DB | Pinecone (Serverless – AWS us-east-1) |
| Graph DB | Neo4j 5+ |
| LLM | OpenAI GPT-4o-mini |
| Dependencies | `openai`, `pinecone-client`, `neo4j`, `tqdm`, `python-dotenv` |

---

## ⚙️ Setup Instructions  

### 1️⃣ Clone the Repository
`git clone https://github.com/kowshik-04/BlueEnigma_Challenge.git`

`cd BlueEnigma_HybridAI`

### 2️⃣ Create a Virtual Environment
`python -m venv venv`

`source venv/bin/activate`    # macOS/Linux

`venv\Scripts\activate`      # Windows


### 3️⃣ Install Dependencies
`pip install -r requirements.txt`

### 4️⃣ Configure Environment Variables
Create a .env file in the root directory:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

OPENAI_API_KEY=sk-xxxxxx

PINECONE_API_KEY=pcsk_xxxxxx
PINECONE_ENV=us-east-1
PINECONE_INDEX_NAME=vietnam-travel
PINECONE_VECTOR_DIM=1536
```

## 🧩 How to Run
- Step 1: Load Data into Neo4j
`python load_to_neo4j.py`

✅ Loads dataset (vietnam_travel_dataset.json) as nodes and relationships.

- Step 2: Visualize Graph
`python visualize_graph.py`

✅ Generates graph visualization (neo4j_viz.html).

- Step 3: Upload Embeddings to Pinecone
`python pinecone_upload.py`

✅ Embeds and uploads all items as vectors.

- Step 4: Run Hybrid Chat
`python hybrid_chat.py`

✅ Start interacting:

- Enter your travel question: create a romantic 4 day itinerary for Vietnam
💬 Example Output

🔍 Retrieved 5 vector matches.
🌐 Retrieved 19 graph relations.

📘 Top Semantic Matches:
 - Da Lat (Southern Vietnam)
 - Hoi An (Central Vietnam)
 - Phu Quoc Island (Southern Vietnam)

## 🤖 Assistant:
Day 1 – Explore Hoi An’s lantern-lit streets...
Day 2 – Visit Da Lat’s waterfalls...
Day 3 – Relax at Phu Quoc Beach...
Day 4 – Sunset dinner cruise before departure.

## 🧠 Design Highlights
Retry & Error Handling – Handles OpenAI rate limits gracefully

Caching – LRU cache for faster embedding retrieval

Serverless Pinecone Integration – AWS us-east-1 setup for scalability

Optimized Prompts – Chain-of-context design for coherent reasoning

Interactive CLI – Clear formatting and debug transparency

## ⚙️ Scalability
To handle 1M+ nodes:

Use asynchronous batch upserts for Pinecone

Deploy clustered Neo4j AuraDS for distributed graph storage

Add Redis caching for frequently used embeddings

Containerize with Docker + Kubernetes for horizontal scaling

## ⚡ Failure Modes and Mitigation
Failure Mode	Cause	Mitigation
Semantic Drift	Outdated embeddings	Re-embedding & versioning
Graph Gaps	Missing links	Data enrichment
Ranking Imbalance	Weight bias	Hybrid scoring
Latency	Dual queries	Async + caching
API Downtime	External service issue	Retry + fallback

## 🧩 Forward Compatibility
Abstraction layer (VectorDBClient) wraps Pinecone interactions

Config-based endpoints and SDK versioning

Graceful fallbacks for API errors

Modular design for easy SDK migration

## 💡 Reflection
This project builds on my prior CVE Analyzer, where I implemented chunking and semantic retrieval for security insights.
Here, I applied the same hybrid reasoning framework to travel data—combining unstructured semantics with structured graph logic.

It reinforced my belief that the future of AI lies in systems that merge meaning with relationships, enabling reasoning that goes beyond retrieval.

## 🧾 Deliverables
File	Description
pinecone_upload.py	Uploads and indexes vector embeddings
load_to_neo4j.py	Loads graph data
visualize_graph.py	Creates relationship visualization
hybrid_chat.py	Hybrid chat system
improvements.md	Enhancements & reasoning write-up
README.md	Documentation (this file)
chat_demo.png	Screenshot of working chat
pinecone_upsert.png	Pinecone upload proof

## 🏁 Conclusion
This system demonstrates how semantic retrieval and graph reasoning can coexist to form a truly intelligent AI assistant.
It doesn’t just find answers — it understands them.
Whether mapping travel plans or analyzing code vulnerabilities, hybrid reasoning is the bridge between data and true understanding.

## 📧 Contact:
Mente Rama Naga Kowshik
- ✉️ [2200031960cseh@gmail.com]
- 💼 GitHub: github.com/kowshik-04
- 🌐 LinkedIn: linkedin.com/in/kowshik-04
