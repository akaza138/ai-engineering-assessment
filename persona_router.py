"""
Phase 1: Vector-Based Persona Matching (The Router)
====================================================
Uses sentence-transformers + an in-memory cosine-similarity store to embed
bot personas and route incoming posts to the bots most likely to care about
them.

Note: ChromaDB is incompatible with Python 3.14, so this implementation
uses a lightweight pure-Python in-memory vector store backed by numpy.
"""

import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# Bot Persona Definitions
# ─────────────────────────────────────────────

BOT_PERSONAS: Dict[str, str] = {
    "bot_a": (
        "I believe AI and crypto will solve all human problems. "
        "I am highly optimistic about technology, Elon Musk, and space exploration. "
        "I dismiss regulatory concerns."
    ),
    "bot_b": (
        "I believe late-stage capitalism and tech monopolies are destroying society. "
        "I am highly critical of AI, social media, and billionaires. "
        "I value privacy and nature."
    ),
    "bot_c": (
        "I strictly care about markets, interest rates, trading algorithms, "
        "and making money. I speak in finance jargon and view everything "
        "through the lens of ROI."
    ),
}

BOT_NAMES: Dict[str, str] = {
    "bot_a": "Tech Maximalist",
    "bot_b": "Doomer / Skeptic",
    "bot_c": "Finance Bro",
}


# ─────────────────────────────────────────────
# Lightweight In-Memory Vector Store
# ─────────────────────────────────────────────

class InMemoryVectorStore:
    """
    A simple in-memory cosine-similarity vector store backed by numpy.
    Replaces ChromaDB for Python 3.14 compatibility.
    """

    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.embeddings: np.ndarray | None = None

    def add(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self.embeddings = self.model.encode(documents, normalize_embeddings=True)

    def query(self, query_texts: List[str], n_results: int = 3) -> Dict:
        """Return top-n results by cosine similarity (higher = more similar)."""
        assert self.embeddings is not None, "Store is empty — call .add() first."
        query_emb = self.model.encode(query_texts, normalize_embeddings=True)  # (1, dim)

        # Cosine similarity: dot product of L2-normalised vectors
        sims = (query_emb @ self.embeddings.T)  # (1, n_docs)

        # Convert similarity → distance (to match original ChromaDB interface)
        dists = 1.0 - sims  # (1, n_docs)

        results_ids, results_docs, results_meta, results_dists = [], [], [], []
        for q_dists in dists:
            top_idx = np.argsort(q_dists)[:n_results]
            results_ids.append([self.ids[i] for i in top_idx])
            results_docs.append([self.documents[i] for i in top_idx])
            results_meta.append([self.metadatas[i] for i in top_idx])
            results_dists.append([float(q_dists[i]) for i in top_idx])

        return {
            "ids":       results_ids,
            "documents": results_docs,
            "metadatas": results_meta,
            "distances": results_dists,
        }


# ─────────────────────────────────────────────
# Vector Store Setup
# ─────────────────────────────────────────────

def build_persona_vector_store() -> Tuple[InMemoryVectorStore, SentenceTransformer]:
    """
    Initialises an in-memory vector store and stores each bot persona
    as an embedded document.

    Returns:
        collection  – the populated InMemoryVectorStore
        model       – the SentenceTransformer model (reused for query embedding)
    """
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    print(f"[Phase 1] Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    print("[Phase 1] Initialising in-memory vector store ...")
    store = InMemoryVectorStore(embedding_model=model)

    store.add(
        ids=list(BOT_PERSONAS.keys()),
        documents=list(BOT_PERSONAS.values()),
        metadatas=[{"name": BOT_NAMES[k]} for k in BOT_PERSONAS],
    )

    print(f"[Phase 1] Stored {len(BOT_PERSONAS)} bot personas in vector store.\n")
    return store, model


# ─────────────────────────────────────────────
# Core Routing Function
# ─────────────────────────────────────────────

def route_post_to_bots(
    post_content: str,
    collection: InMemoryVectorStore,
    threshold: float = 0.35,
    top_k: int = 3,
) -> List[Dict]:
    """
    Embed the incoming post and return only the bots whose persona
    cosine *similarity* exceeds `threshold`.

    The store returns cosine *distance* (1 - similarity), so we convert:
        similarity = 1 - distance

    Args:
        post_content : raw text of the social-media post
        collection   : populated InMemoryVectorStore
        threshold    : minimum cosine similarity to include a bot (default 0.35)
        top_k        : how many nearest neighbours to retrieve before filtering

    Returns:
        List of matched bots, each a dict with keys:
            bot_id, name, persona, similarity
    """
    results = collection.query(
        query_texts=[post_content],
        n_results=top_k,
    )

    matched_bots: List[Dict] = []
    ids        = results["ids"][0]
    distances  = results["distances"][0]
    metadatas  = results["metadatas"][0]
    documents  = results["documents"][0]

    for bot_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
        similarity = 1.0 - dist          # convert distance → similarity
        if similarity >= threshold:
            matched_bots.append({
                "bot_id":     bot_id,
                "name":       meta["name"],
                "persona":    doc,
                "similarity": round(similarity, 4),
            })

    return matched_bots


# ─────────────────────────────────────────────
# Demo / Entry Point
# ─────────────────────────────────────────────

def run_phase1_demo():
    collection, _ = build_persona_vector_store()

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits a new all-time high – should you buy the dip?",
        "Big Tech is buying up senators. Democracy is dead.",
        "The Fed raised interest rates again. Bond yields are spiking.",
    ]

    print("=" * 65)
    print("PHASE 1 — Persona Router Demo")
    print("=" * 65)

    for post in test_posts:
        print(f'\n POST: "{post}"')
        matched = route_post_to_bots(post, collection, threshold=0.25)

        if matched:
            print("  Matched bots:")
            for bot in matched:
                print(f"     * [{bot['bot_id'].upper()}] {bot['name']}  "
                      f"(similarity={bot['similarity']:.4f})")
        else:
            print("  No bot matched above threshold.")

    print("\n" + "=" * 65)
    return collection


if __name__ == "__main__":
    run_phase1_demo()
