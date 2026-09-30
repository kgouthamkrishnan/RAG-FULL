import os
import logging
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_backend")

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

class Config:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    EMBED_MODEL: str    = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL: str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CHUNK_SIZE: int     = 600
    CHUNK_OVERLAP: int  = 80
    BM25_TOP_K: int     = 8
    DENSE_TOP_K: int    = 8
    RERANK_TOP_K: int   = 4
    MAX_TOKENS: int     = 2048

    @classmethod
    def validate(cls):
        if not cls.GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY is not set in .env")

Config.validate()

# ── Singletons ────────────────────────────────────────────────────────────────
logger.info("Loading embedding model: %s", Config.EMBED_MODEL)
embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBED_MODEL)

logger.info("Loading cross-encoder re-ranker: %s", Config.RERANK_MODEL)
cross_encoder = CrossEncoder(Config.RERANK_MODEL)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.CHUNK_SIZE,
    chunk_overlap=Config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Global state
vector_db: FAISS | None = None
all_chunks: list[Document] = []

# ── Indexing ──────────────────────────────────────────────────────────────────
def index_documents(documents: list[Document]) -> int:
    global vector_db, all_chunks

    if not documents:
        raise ValueError("No documents provided for indexing.")

    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Document produced no usable text chunks.")

    logger.info("Indexing %d chunks...", len(chunks))
    vector_db = FAISS.from_documents(chunks, embedding_model)
    all_chunks = chunks
    logger.info("Vector DB + BM25 ready — %d chunks indexed.", len(chunks))
    return len(chunks)

# ── LLM Helper ────────────────────────────────────────────────────────────────
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GEMINI_API_KEY,
        max_output_tokens=Config.MAX_TOKENS,
        temperature=0.3,
    )

# ── Step 1: Query Rewriting ───────────────────────────────────────────────────
def rewrite_query(query: str, llm: ChatGoogleGenerativeAI) -> str:
    prompt = (
        "You are a search query optimizer. Rewrite the following user question "
        "into a clear, detailed search query to find the most relevant information "
        "in a document. One sentence only. Do NOT answer — just rewrite.\n\n"
        f"User question: {query}\n\n"
        "Optimized search query:"
    )
    result = llm.invoke(prompt)
    rewritten = result.content.strip()
    logger.info("Query rewritten: %r → %r", query[:50], rewritten[:50])
    return rewritten

# ── Step 2: Hybrid Retrieval (BM25 + Dense) ───────────────────────────────────
def hybrid_retrieve(query: str) -> list[Document]:
    """Manual hybrid fusion — no EnsembleRetriever dependency needed."""

    # BM25 sparse retrieval
    bm25 = BM25Retriever.from_documents(all_chunks)
    bm25.k = Config.BM25_TOP_K
    bm25_docs = bm25.invoke(query)

    # Dense semantic retrieval
    dense_docs = vector_db.similarity_search(query, k=Config.DENSE_TOP_K)

    # Merge — deduplicate by content, preserve order (BM25 first, then dense)
    seen = set()
    merged = []
    for doc in bm25_docs + dense_docs:
        key = doc.page_content[:100]  # first 100 chars as fingerprint
        if key not in seen:
            seen.add(key)
            merged.append(doc)

    logger.info(
        "Hybrid retrieval — BM25: %d, Dense: %d, Merged: %d",
        len(bm25_docs), len(dense_docs), len(merged)
    )
    return merged

# ── Step 3: Re-ranking (CrossEncoder) ────────────────────────────────────────
def rerank_docs(query: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:Config.RERANK_TOP_K]]

    logger.info("Re-ranked — top %d docs selected.", len(top_docs))
    return top_docs

# ── Step 4: Context Compression (LLM filters noise) ──────────────────────────
def compress_context(query: str, docs: list[Document], llm: ChatGoogleGenerativeAI) -> str:
    raw_context = "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = (
        "You are a context filter. From the document excerpts below, extract ONLY "
        "the sentences and paragraphs that are directly relevant to answering the question. "
        "Remove irrelevant content. Keep exact wording — do not paraphrase.\n\n"
        f"Question: {query}\n\n"
        f"Document excerpts:\n{raw_context}\n\n"
        "Relevant content only:"
    )

    result = llm.invoke(prompt)
    compressed = result.content.strip()
    logger.info("Context compressed: %d chars → %d chars", len(raw_context), len(compressed))
    return compressed

# ── Step 5: Final Answer Generation ──────────────────────────────────────────
def generate_answer(
    query: str,
    context: str,
    chat_history: list[dict],
    llm: ChatGoogleGenerativeAI,
) -> str:
    history_str = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['text']}\n"

    prompt = f"""You are an intelligent document assistant. Answer accurately based strictly on the provided context.

DOCUMENT CONTEXT:
{context}

{"CONVERSATION HISTORY:" + chr(10) + history_str if history_str else ""}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer strictly from the document context
- For rules, regulations, or procedures — explain clearly in simple language
- Use **bold** for important terms or section headings
- Use bullet points or numbered lists where appropriate
- Break into short paragraphs — never one long block
- If something is not in the document, clearly say so
- Be thorough yet concise

ANSWER:"""

    response = llm.invoke(prompt)
    return response.content

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def query_rag(query_text: str, chat_history: list[dict] | None = None) -> str:
    """
    Full Agentic RAG Pipeline:
    1. Query Rewriting
    2. Hybrid Retrieval (BM25 + Dense FAISS)
    3. Cross-Encoder Re-ranking
    4. LLM Context Compression
    5. Answer Generation with chat memory
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text cannot be empty.")

    if vector_db is None or not all_chunks:
        return "No document indexed yet. Please upload a file first."

    logger.info("=== RAG Pipeline Start ===")
    llm = get_llm()

    # Step 1 — Query Rewriting
    rewritten_query = rewrite_query(query_text, llm)

    # Step 2 — Hybrid Retrieval
    retrieved_docs = hybrid_retrieve(rewritten_query)

    # Step 3 — Re-ranking
    reranked_docs = rerank_docs(rewritten_query, retrieved_docs)

    if not reranked_docs:
        return "No relevant content found in the document for your query."

    # Step 4 — Context Compression
    compressed_context = compress_context(query_text, reranked_docs, llm)

    # Step 5 — Answer Generation
    answer = generate_answer(query_text, compressed_context, chat_history or [], llm)

    logger.info("=== RAG Pipeline Complete ===")
    return answer