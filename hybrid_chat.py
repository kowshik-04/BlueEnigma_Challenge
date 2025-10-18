# # hybrid_chat.py
# import json
# from typing import List
# from openai import OpenAI
# from pinecone import Pinecone, ServerlessSpec
# from neo4j import GraphDatabase
# import config

# # -----------------------------
# # Config
# # -----------------------------
# EMBED_MODEL = "text-embedding-3-small"
# CHAT_MODEL = "gpt-4o-mini"
# TOP_K = 5

# INDEX_NAME = config.PINECONE_INDEX_NAME

# # -----------------------------
# # Initialize clients
# # -----------------------------
# client = OpenAI(api_key=config.OPENAI_API_KEY)
# pc = Pinecone(api_key=config.PINECONE_API_KEY)

# # Connect to Pinecone index
# if INDEX_NAME not in pc.list_indexes().names():
#     print(f"Creating managed index: {INDEX_NAME}")
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=config.PINECONE_VECTOR_DIM,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

# index = pc.Index(INDEX_NAME)

# # Connect to Neo4j
# driver = GraphDatabase.driver(
#     config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
# )

# # -----------------------------
# # Helper functions
# # -----------------------------
# from openai import OpenAIError

# def embed_text(text: str) -> List[float]:
#     try:
#         resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
#         return resp.data[0].embedding
#     except OpenAIError as e:
#         print(f"⚠️ Embedding error: {e}")
#         return [0.0] * config.PINECONE_VECTOR_DIM  # dummy vector to continue


# def pinecone_query(query_text: str, top_k=TOP_K):
#     """Query Pinecone index using embedding."""
#     vec = embed_text(query_text)
#     res = index.query(
#         vector=vec,
#         top_k=top_k,
#         include_metadata=True,
#         include_values=False
#     )
#     print("DEBUG: Pinecone top 5 results:")
#     print(len(res.matches))
#     return res.matches

# def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
#     """Fetch neighboring nodes from Neo4j."""
#     facts = []
#     with driver.session() as session:
#         for nid in node_ids:
#             q = (
#                 "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
#                 "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
#                 "m.name AS name, m.type AS type, m.description AS description "
#                 "LIMIT 10"
#             )
#             recs = session.run(q, nid=nid)
#             for r in recs:
#                 facts.append({
#                     "source": nid,
#                     "rel": r["rel"],
#                     "target_id": r["id"],
#                     "target_name": r["name"],
#                     "target_desc": (r["description"] or "")[:400],
#                     "labels": r["labels"]
#                 })
#     print("DEBUG: Graph facts:")
#     print(len(facts))
#     return facts

# def build_prompt(user_query, pinecone_matches, graph_facts):
#     """Build a chat prompt combining vector DB matches and graph facts."""
#     system = (
#         "You are a helpful travel assistant. Use the provided semantic search results "
#         "and graph facts to answer the user's query briefly and concisely. "
#         "Cite node ids when referencing specific places or attractions."
#     )

#     vec_context = []
#     for m in pinecone_matches:
#         meta = m["metadata"]
#         score = m.get("score", None)
#         snippet = f"- id: {m['id']}, name: {meta.get('name','')}, type: {meta.get('type','')}, score: {score}"
#         if meta.get("city"):
#             snippet += f", city: {meta.get('city')}"
#         vec_context.append(snippet)

#     graph_context = [
#         f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
#         for f in graph_facts
#     ]

#     prompt = [
#         {"role": "system", "content": system},
#         {"role": "user", "content":
#          f"User query: {user_query}\n\n"
#          "Top semantic matches (from vector DB):\n" + "\n".join(vec_context[:10]) + "\n\n"
#          "Graph facts (neighboring relations):\n" + "\n".join(graph_context[:20]) + "\n\n"
#          "Based on the above, answer the user's question. If helpful, suggest 2–3 concrete itinerary steps or tips and mention node ids for references."}
#     ]
#     return prompt

# def call_chat(prompt_messages):
#     """Call OpenAI ChatCompletion."""
#     resp = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=prompt_messages,
#         max_tokens=600,
#         temperature=0.2
#     )
#     return resp.choices[0].message.content

# # -----------------------------
# # Interactive chat
# # -----------------------------
# def interactive_chat():
#     print("Hybrid travel assistant. Type 'exit' to quit.")
#     while True:
#         query = input("\nEnter your travel question: ").strip()
#         if not query or query.lower() in ("exit","quit"):
#             break

#         matches = pinecone_query(query, top_k=TOP_K)
#         match_ids = [m["id"] for m in matches]
#         graph_facts = fetch_graph_context(match_ids)
#         prompt = build_prompt(query, matches, graph_facts)
#         answer = call_chat(prompt)
#         print("\n=== Assistant Answer ===\n")
#         print(answer)
#         print("\n=== End ===\n")

# if __name__ == "__main__":
#     interactive_chat()



# hybrid_chat.py
"""
Blue Enigma Labs — Hybrid AI Travel Assistant
Combines Pinecone (vector search) + Neo4j (graph context) + OpenAI (reasoning).
"""

import textwrap
import time
from typing import List
from functools import lru_cache

from openai import OpenAI, OpenAIError
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config


# -----------------------------
# 🔧 Configuration
# -----------------------------
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
RATE_LIMIT_SLEEP = 3       # seconds between retries if rate-limited

INDEX_NAME = config.PINECONE_INDEX_NAME


# -----------------------------
# 🚀 Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ matches your account
    )

index = pc.Index(INDEX_NAME)

# Neo4j driver
driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)


# -----------------------------
# 🧠 Helper functions
# -----------------------------
@lru_cache(maxsize=256)
def embed_text_cached(text: str) -> List[float]:
    """Get (cached) embedding for text."""
    for _ in range(3):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
            return resp.data[0].embedding
        except OpenAIError as e:
            print(f"⚠️  Embedding error: {e}. Retrying...")
            time.sleep(RATE_LIMIT_SLEEP)
    print("❌ Failed to get embedding; returning zero-vector.")
    return [0.0] * config.PINECONE_VECTOR_DIM


def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone using embedding."""
    vec = embed_text_cached(query_text)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    matches = res.matches
    print(f"🔍 Retrieved {len(matches)} vector matches.")
    return matches


def fetch_graph_context(node_ids: List[str]):
    """Fetch neighboring nodes from Neo4j for each Pinecone id."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, "
                "m.id AS id, m.name AS name, m.description AS description "
                "LIMIT 8"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:200],
                    "labels": r["labels"]
                })
    print(f"🌐 Retrieved {len(facts)} graph relations.")
    return facts


def build_prompt(user_query, pinecone_matches, graph_facts):
    """Construct an optimal chat prompt."""
    system = (
        "You are an expert AI travel planner. "
        "Use the following semantic search results and graph relationships "
        "to answer the user query precisely. "
        "If helpful, propose a short itinerary or list of attractions. "
        "Cite node ids when referencing locations."
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m.metadata
        snippet = f"- id: {m.id}, name: {meta.get('name','')}, city: {meta.get('city','')}, type: {meta.get('type','')}"
        vec_context.append(snippet)

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"User query: {user_query}\n\n"
         f"Vector context (semantic matches):\n{chr(10).join(vec_context[:10])}\n\n"
         f"Graph context (related nodes):\n{chr(10).join(graph_context[:20])}\n\n"
         "Generate an intelligent, concise response based on both contexts."}
    ]
    return prompt


def call_chat(prompt_messages):
    """Call OpenAI ChatCompletion with retry on rate limits."""
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=prompt_messages,
                max_tokens=600,
                temperature=0.3
            )
            return resp.choices[0].message.content.strip()
        except OpenAIError as e:
            print(f"⚠️  Chat error: {e}. Retrying...")
            time.sleep(RATE_LIMIT_SLEEP)
    return "❌ Failed to get response from model."


# -----------------------------
# 💬 Interactive console chat
# -----------------------------
def interactive_chat():
    print("=" * 80)
    print("🧭  Blue Enigma Hybrid AI Travel Assistant")
    print("Type 'exit' or 'quit' to leave.")
    print("=" * 80)

    while True:
        query = input("\n❓ Enter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break

        # 1️⃣ Semantic retrieval
        matches = pinecone_query(query)
        match_ids = [m.id for m in matches]

        # 2️⃣ Graph retrieval
        graph_facts = fetch_graph_context(match_ids)

        # 3️⃣ Display top matches
        print("\n📘 Top Semantic Matches:")
        for m in matches[:3]:
            meta = m.metadata
            print(f" • {meta.get('name','')} ({meta.get('city','')}) — score {m.score:.3f}")

        # 4️⃣ Build prompt and reason
        prompt = build_prompt(query, matches, graph_facts)
        answer = call_chat(prompt)

        # 5️⃣ Display nicely
        print("\n" + "=" * 80)
        print("🤖  Assistant:\n")
        print(textwrap.fill(answer, width=100))
        print("=" * 80)

    driver.close()
    print("\n👋 Session ended. Safe travels!\n")


# -----------------------------
if __name__ == "__main__":
    interactive_chat()
