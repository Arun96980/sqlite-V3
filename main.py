import os
import re
import json
import time
import argparse
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import asyncio
import aiosqlite  # pip install aiosqlite
import aiohttp    # pip install aiohttp
from datetime import datetime
from collections import defaultdict
from tqdm.asyncio import tqdm # Use tqdm's async version for async loops
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor # Keep for CPU-bound tasks if needed, but Tika is I/O
from sklearn.feature_extraction.text import TfidfVectorizer
import pysbd
from sklearn.metrics.pairwise import cosine_similarity
from itertools import islice
from fastapi import FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any  # Add this with other imports

# -----------------------------
# Configuration & Constants
# -----------------------------
# Consider moving more settings to a config file (YAML/JSON) or more args
DEFAULT_MODEL = "intfloat/e5-large"
DEFAULT_TIKA_URL = "http://localhost:9998/rmeta/text"
DEFAULT_DB_PATH = "resume_engine.db"
DEFAULT_INDEX_PATH = "faiss.index"
DEFAULT_FEEDBACK_PATH = "search_feedback.json" # Keep feedback simple for now
DEFAULT_FAISS_CANDIDATES = 80
DEFAULT_TOP_K = 5
HYBRID_ALPHA = 0.7 # Weight for semantic score in RAGRetriever hybrid score
RRF_K = 60 # Constant for Reciprocal Rank Fusion
API_PORT = 8000
API_HOST = "0.0.0.0"

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
    handlers=[
        RotatingFileHandler("resume_parser_optimized.log", encoding='utf-8', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add UTF-8 encoding for Windows compatibility
if os.name == 'nt':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
# -----------------------------
# Database Setup & Operations
# -----------------------------
async def initialize_database(db_path):
    """Creates necessary tables in the SQLite database if they don't exist."""
    async with aiosqlite.connect(db_path) as db:
        # Metadata table: faiss_id corresponds to the vector's position in the index
        await db.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                faiss_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_hash TEXT UNIQUE NOT NULL,
                file TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                sentence TEXT NOT NULL,
                length INTEGER NOT NULL
            )
        """)
        # Index on sentence_hash for faster lookups
        await db.execute("CREATE INDEX IF NOT EXISTS idx_sentence_hash ON metadata(sentence_hash)")

        # LLM Cache table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                score REAL,
                explanation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
    logger.info(f"Database initialized at {db_path}")

async def get_processed_sentence_hashes(db_path):
    """Fetches all existing sentence hashes from the database."""
    hashes = set()
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT sentence_hash FROM metadata") as cursor:
                async for row in cursor:
                    hashes.add(row[0])
    except Exception as e:
        logger.error(f"Error fetching processed sentence hashes: {e}")
    return hashes

async def add_metadata_batch(db_path, metadata_batch):
    """Adds a batch of new metadata entries to the database."""
    if not metadata_batch:
        return 0
    sql = "INSERT OR IGNORE INTO metadata (sentence_hash, file, file_hash, sentence, length) VALUES (?, ?, ?, ?, ?)"
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.executemany(sql, [(m['sentence_hash'], m['file'], m['file_hash'], m['sentence'], m['length']) for m in metadata_batch])
            await db.commit()
            # Get the number of rows actually inserted (executemany doesn't return it directly with IGNORE)
            # This is an approximation; a more accurate way might involve checking hashes before inserting
            logger.info(f"Attempted to add {len(metadata_batch)} metadata entries.")
            # To get exact count, might need to SELECT count before/after or use different logic
            return len(metadata_batch) # Return attempted count
    except Exception as e:
        logger.error(f"Error adding metadata batch: {e}")
        return 0



def chunked(iterable, size):
    """Yield successive chunks from iterable of given size."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

async def get_metadata_by_faiss_ids(db_path, faiss_ids):
    """Retrieves metadata for specific FAISS IDs in chunks to avoid SQLite variable limits."""
    if not faiss_ids:
        return []

    results = []
    try:
        async with aiosqlite.connect(db_path) as db:
            for chunk in chunked(faiss_ids, 900):  # Stay under SQLite's 999 param limit
                placeholders = ','.join('?' for _ in chunk)
                sql = f"""
                    SELECT faiss_id, sentence_hash, file, file_hash, sentence, length
                    FROM metadata
                    WHERE faiss_id IN ({placeholders})
                """
                async with db.execute(sql, chunk) as cursor:
                    async for row in cursor:
                        results.append({
                            "faiss_id": row[0], "sentence_hash": row[1], "file": row[2],
                            "file_hash": row[3], "sentence": row[4], "length": row[5]
                        })
    except Exception as e:
        logger.error(f"Error fetching metadata by FAISS IDs: {e}")
    return results

async def get_all_metadata_for_tfidf(db_path):
    """Retrieves all necessary metadata for TF-IDF fitting."""
    results = []
    try:
        async with aiosqlite.connect(db_path) as db:
            # Only select columns needed for TF-IDF and result construction
            async with db.execute("SELECT faiss_id, sentence_hash, file, sentence FROM metadata ORDER BY faiss_id") as cursor:
                 async for row in cursor:
                    results.append({
                        "faiss_id": row[0], "sentence_hash": row[1],
                        "file": row[2], "sentence": row[3]
                    })
    except Exception as e:
        logger.error(f"Error fetching all metadata for TF-IDF: {e}")
    return results

async def get_from_cache(db_path, cache_key):
    """Retrieves an item from the LLM cache."""
    try:
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT score, explanation FROM llm_cache WHERE cache_key = ?", (cache_key,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {"score": row[0], "explanation": row[1]}
    except Exception as e:
        logger.error(f"Error getting from cache (key: {cache_key}): {e}")
    return None

async def store_to_cache(db_path, cache_key, score, explanation):
    """Stores or updates an item in the LLM cache."""
    sql = """
        INSERT INTO llm_cache (cache_key, score, explanation, timestamp) VALUES (?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            score = excluded.score,
            explanation = excluded.explanation,
            timestamp = excluded.timestamp
    """
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(sql, (cache_key, score, explanation, datetime.now()))
            await db.commit()
    except Exception as e:
        logger.error(f"Error storing to cache (key: {cache_key}): {e}")

# -----------------------------
# Custom Sentence Splitting with pysbd
# -----------------------------
def split_sentences_with_pysbd(text):
    """Split text into meaningful sentences using pysbd with custom merging."""
    # This function remains synchronous as pysbd is likely CPU-bound
    try:
        segmenter = pysbd.Segmenter(language="en", clean=False)
        raw_sentences = segmenter.segment(text)
    except Exception as e:
        logger.error(f"Error during sentence segmentation: {e}")
        return []

    meaningful = []
    buffer = ""
    for sent in raw_sentences:
        # Clean unwanted bullet characters and extra spaces.
        sent_clean = re.sub(r"[\u2022\u25AA\u25CF\u25A0•▪➢]", " ", sent)
        sent_clean = re.sub(r"\s+", " ", sent_clean).strip()

        # Skip empty sentences after cleaning
        if not sent_clean:
            continue

        # Combine short sentences (likely list items or fragments)
        # Adjust the threshold '5' as needed
        if len(sent_clean.split()) < 5:
            buffer += " " + sent_clean
        else:
            if buffer:
                sent_clean = (buffer.strip() + " " + sent_clean).strip()
                buffer = ""
            meaningful.append(sent_clean)

    # Add any remaining buffer content as a final sentence
    if buffer:
        meaningful.append(buffer.strip())

    return meaningful

# -----------------------------
# Resume Processing & Metadata Generation
# -----------------------------
def compute_file_hash(file_path):
    """Computes SHA256 hash for a file."""
    # Synchronous, potentially CPU-bound for large files
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error computing hash for {file_path}: {e}")
        return None

def compute_sentence_hash(sentence):
    """Computes SHA1 hash for a sentence (faster than SHA256, sufficient for uniqueness)."""
    return hashlib.sha1(sentence.encode("utf-8")).hexdigest()

async def extract_clean_text_async(file_path, tika_url, session):
    """Extract text using Apache Tika asynchronously."""
    try:
        with open(file_path, "rb") as f:
            file_data = f.read() # Read file content once

        headers = {"Accept": "application/json", "Content-Type": "application/octet-stream"} # Tika prefers octet-stream
        async with session.put(tika_url, data=file_data, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as response: # Added timeout
            response.raise_for_status() # Raise exception for bad status codes
            content_json = await response.json()
            if content_json and isinstance(content_json, list) and "X-TIKA:content" in content_json[0]:
                 # Basic cleaning: replace multiple newlines/spaces
                text = content_json[0]["X-TIKA:content"]
                text = re.sub(r'\s*\n\s*', '\n', text) # Consolidate newlines
                text = re.sub(r'[ \t]+', ' ', text) # Consolidate spaces/tabs
                return text.strip()
            else:
                logger.warning(f"Tika returned empty or unexpected content for {file_path}: {content_json}")
                return ""
    except aiohttp.ClientError as e:
        logger.error(f"❌ Network error extracting from {file_path} with Tika: {e}")
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
    except Exception as e:
        logger.error(f"❌ Unexpected error extracting from {file_path}: {e} (Type: {type(e)})")
    return ""

async def process_file(file_info, model, tika_url, existing_sentence_hashes, session):
    """Processes a single file: extract text, split, hash, filter, embed."""
    pdf_dir, file = file_info
    file_path = os.path.join(pdf_dir, file)
    logger.debug(f"Processing file: {file_path}")

    file_hash = compute_file_hash(file_path)
    if not file_hash:
        return [], [] # Return empty lists if hashing failed

    # Check if file hash exists? (Could add optimization to skip files if hash hasn't changed, requires storing file hashes)

    text = await extract_clean_text_async(file_path, tika_url, session)
    if not text:
        logger.warning(f"No text extracted from {file}")
        return [], []

    sentences = split_sentences_with_pysbd(text)
    if not sentences:
        logger.warning(f"No sentences obtained from {file}")
        return [], []

    new_metadata = []
    sentences_to_embed = []

    for sentence in sentences:
        # Basic check for meaningful content length
        if len(sentence) < 20 or len(sentence.split()) < 5: # Increased minimum length slightly
             continue

        sent_hash = compute_sentence_hash(sentence)
        if sent_hash in existing_sentence_hashes:
            continue # Skip already processed sentences

        existing_sentence_hashes.add(sent_hash) # Add locally to prevent duplicates within the same run

        metadata = {
            "file": file,
            "file_hash": file_hash,
            "sentence": sentence,
            "length": len(sentence),
            "sentence_hash": sent_hash
            # Embedding will be added later
        }
        new_metadata.append(metadata)
        sentences_to_embed.append(f"passage: {sentence}") # Add prefix for e5 model

    if sentences_to_embed:
        try:
            # SentenceTransformer's encode is CPU/GPU bound, run in executor
            # Or ensure SentenceTransformer uses GPU if available
            loop = asyncio.get_running_loop()
            # Note: show_progress_bar=True might not work well with run_in_executor
            embeddings = await loop.run_in_executor(
                None, # Use default ThreadPoolExecutor
                lambda:model.encode(sentences_to_embed,
                normalize_embeddings=True)
                
                # show_progress_bar=True # Might cause issues in executor
            )
            logger.info(f"Generated {len(embeddings)} embeddings for {file}")
            # Add embeddings back to metadata
            vectors = [emb.astype(np.float32) for emb in embeddings] # Ensure float32
            return new_metadata, vectors
        except Exception as e:
            logger.error(f"Error generating embeddings for {file}: {e}")
            return [], [] # Return empty if embedding fails
    else:
        return [], [] # No new sentences found

async def process_all_files(pdf_dir, model, tika_url, db_path):
    """Processes all supported files in a directory asynchronously."""
    supported_exts = ('.pdf', '.doc', '.docx', '.pptx', '.txt', '.rtf')
    try:
        files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(supported_exts) and os.path.isfile(os.path.join(pdf_dir, f))]
    except FileNotFoundError:
        logger.error(f"Directory not found: {pdf_dir}")
        return [], []
    except Exception as e:
        logger.error(f"Error listing files in {pdf_dir}: {e}")
        return [], []

    if not files:
        logger.info("📂 No supported files found.")
        return [], []

    logger.info(f"Found {len(files)} supported files.")
    existing_sentence_hashes = await get_processed_sentence_hashes(db_path)
    logger.info(f"Found {len(existing_sentence_hashes)} existing sentence hashes in DB.")

    all_new_metadata = []
    all_new_vectors = []

    # Use aiohttp ClientSession for connection pooling
    async with aiohttp.ClientSession() as session:
        # Create tasks for each file
        tasks = [process_file((pdf_dir, file), model, tika_url, existing_sentence_hashes, session) for file in files]

        # Process tasks concurrently with progress bar
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="🔁 Processing Files"):
            try:
                file_metadata, file_vectors = await future
                if file_metadata:
                    all_new_metadata.extend(file_metadata)
                    all_new_vectors.extend(file_vectors)
            except Exception as e:
                logger.error(f"Error processing a file future: {e}")

    logger.info(f"Total new sentences found: {len(all_new_metadata)}")
    return all_new_metadata, all_new_vectors

# -----------------------------
# FAISS Index Building
# -----------------------------
def build_or_load_index(index_path, dim, force_rebuild=False):
    """Loads an existing FAISS index or creates a new one."""
    if os.path.exists(index_path) and not force_rebuild:
        try:
            logger.info(f"Loading existing FAISS index from {index_path}")
            index = faiss.read_index(index_path)
            if index.d != dim:
                 logger.warning(f"Index dimension mismatch! Index dim={index.d}, Model dim={dim}. Rebuilding.")
                 index = faiss.IndexFlatIP(dim) # Recreate if dimensions don't match
            else:
                 logger.info(f"Successfully loaded index with {index.ntotal} vectors.")
            return index
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}. Creating a new one.")
            # Fall through to create a new index
    else:
        logger.info("Creating new FAISS index.")

    # --- Optimization Point ---
    # For large datasets, use IndexIVFFlat or IndexIVFPQ
    # Example for IndexIVFFlat:
    # nlist = 100 # Number of clusters (adjust based on dataset size)
    # quantizer = faiss.IndexFlatIP(dim) # Or IndexFlatL2
    # index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    # # Requires training: index.train(training_vectors)
    # logger.info(f"Created IndexIVFFlat with {nlist} lists.")
    # --------------------------
    index = faiss.IndexFlatIP(dim) # Using Inner Product (cosine similarity on normalized vectors)
    return index

async def build_incremental_index(new_metadata, new_vectors, model, index_path, db_path, force_rebuild=False):
    """Adds new vectors to the FAISS index and saves metadata to DB."""
    if not new_metadata or not new_vectors:
        logger.info("✅ No new sentences to process. Index and DB remain unchanged.")
        return

    # Determine embedding dimension
    try:
        # Get dimension from model or a sample vector
        dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model embedding dimension: {dim}")
    except Exception as e:
         logger.error(f"Could not determine model dimension: {e}. Using dimension of first new vector.")
         if new_vectors:
             dim = new_vectors[0].shape[0]
         else:
             logger.error("Cannot determine dimension, no vectors provided.")
             return # Cannot proceed without dimension

    index = build_or_load_index(index_path, dim, force_rebuild)
    initial_index_size = index.ntotal

    # Add new vectors to FAISS index
    if new_vectors:
        vectors_np = np.array(new_vectors).astype('float32') # Ensure correct dtype
        logger.info(f"Adding {len(vectors_np)} new vectors to FAISS index...")
        try:
            # --- Training Step (Required for IVF indexes) ---
            # if isinstance(index, faiss.IndexIVF) and not index.is_trained:
            #     logger.info("Training IVF index...")
            #     # Need representative training data (e.g., a subset of vectors)
            #     # Assuming 'vectors_np' can be used for training if it's the first batch
            #     index.train(vectors_np)
            #     logger.info("Training complete.")
            # --------------------------------------------------
            index.add(vectors_np)
            logger.info(f"Successfully added vectors. Index size now: {index.ntotal}")
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS index: {e}")
            return # Don't proceed if adding vectors failed

        # Save the updated index
        try:
            faiss.write_index(index, index_path)
            logger.info(f"FAISS index saved to {index_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            # Decide if you want to proceed with metadata update despite index save failure

    # Add metadata to the database
    # Important: Assumes the order of new_metadata matches the order of new_vectors added
    added_count = await add_metadata_batch(db_path, new_metadata)
    logger.info(f"Added {added_count} metadata entries to the database.")

    final_index_size = index.ntotal
    logger.info(f"✅ Indexing complete. Added {final_index_size - initial_index_size} new vectors/metadata entries.")


# -----------------------------
# RAG & Hybrid Search Components
# -----------------------------

def build_contextual_chunks(metadata_list, faiss_ids, window_size=2):
    """Create contextual chunks centered around the retrieved FAISS IDs."""
    # Create a mapping from faiss_id to its index in the metadata_list
    id_to_list_idx = {item['faiss_id']: i for i, item in enumerate(metadata_list)}
    chunks = []

    # Ensure metadata is sorted by faiss_id if not already guaranteed
    metadata_list.sort(key=lambda x: x['faiss_id'])

    for target_faiss_id in faiss_ids:
        if target_faiss_id not in id_to_list_idx:
            logger.warning(f"FAISS ID {target_faiss_id} not found in provided metadata batch.")
            continue

        list_idx = id_to_list_idx[target_faiss_id]
        target_item = metadata_list[list_idx]

        # Determine context window based on list index
        start_idx = max(0, list_idx - window_size)
        end_idx = min(len(metadata_list), list_idx + window_size + 1)

        # Gather sentences within the window
        context_sentences = [metadata_list[j]["sentence"] for j in range(start_idx, end_idx)]
        context = " ".join(context_sentences)

        chunks.append({
            "text": context,
            "source": target_item["file"],
            "sentence_hash": target_item["sentence_hash"], # Hash of the *central* sentence
            "faiss_id": target_faiss_id # ID of the *central* sentence
        })
    return chunks


class RAGRetriever:
    """Handles FAISS search, chunk building, and hybrid scoring."""
    def __init__(self, index_path, db_path, model, faiss_candidates=80, window_size=2):
        self.index_path = index_path
        self.db_path = db_path
        self.model = model
        self.faiss_candidates = faiss_candidates
        self.window_size = window_size
        self.index = None # Load lazily or explicitly
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1) # Added max/min df
        self.tfidf_matrix = None
        self.tfidf_metadata = [] # Metadata corresponding to tfidf_matrix rows

    def load_index(self):
        """Loads the FAISS index."""
        if self.index is None:
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"RAGRetriever loaded FAISS index with {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index in RAGRetriever: {e}")
                raise # Re-raise exception as retriever cannot function without index

    async def fit_tfidf(self):
        """Fits the TF-IDF vectorizer on all metadata."""
        # This might be memory intensive for very large datasets
        logger.info("Fitting TF-IDF vectorizer for RAGRetriever...")
        self.tfidf_metadata = await get_all_metadata_for_tfidf(self.db_path)
        if not self.tfidf_metadata:
             logger.warning("No metadata found to fit TF-IDF.")
             return
        sentences = [m['sentence'] for m in self.tfidf_metadata]
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(sentences)
            logger.info(f"TF-IDF fitted on {len(self.tfidf_metadata)} sentences.")
        except ValueError as e:
             logger.error(f"TF-IDF fitting error (perhaps empty vocabulary?): {e}")
             # Handle case where TF-IDF fitting fails (e.g., only stop words)
             self.tfidf_matrix = None # Ensure matrix is None if fitting failed


    def _hybrid_score(self, query_vec_norm, target_sentence_embedding_norm, target_faiss_id):
        """Calculate hybrid score combining semantic and lexical relevance."""
        # Assumes query_vec_norm and target_sentence_embedding_norm are already normalized
        semantic_score = float(np.dot(query_vec_norm, target_sentence_embedding_norm))

        # Find the row index corresponding to the faiss_id in the TF-IDF matrix
        # This assumes tfidf_metadata is sorted by faiss_id, which get_all_metadata_for_tfidf ensures
        # A mapping might be safer if sorting isn't guaranteed or for performance
        lexical_score = 0.0
        if self.tfidf_matrix is not None and self.tfidf_metadata:
            try:
                # Simple search assuming sorted list; binary search or dict lookup would be faster
                row_idx = -1
                for i, meta in enumerate(self.tfidf_metadata):
                    if meta['faiss_id'] == target_faiss_id:
                        row_idx = i
                        break

                if row_idx != -1:
                    # Get TF-IDF vector for the sentence
                    sentence_tfidf_vector = self.tfidf_matrix[row_idx]
                    # Calculate cosine similarity (or just use sum as a proxy if query isn't transformed)
                    # For simplicity, using sum as a proxy for lexical importance here.
                    # A more correct lexical score would involve transforming the query with the *same* vectorizer
                    # and calculating cosine similarity.
                    lexical_score = sentence_tfidf_vector.sum() # Simple sum proxy
                else:
                     logger.warning(f"Could not find faiss_id {target_faiss_id} in TF-IDF metadata.")

            except Exception as e:
                 logger.error(f"Error calculating lexical score for faiss_id {target_faiss_id}: {e}")


        # Combine scores (adjust alpha weight as needed)
        return HYBRID_ALPHA * semantic_score + (1 - HYBRID_ALPHA) * lexical_score

    async def retrieve(self, query, top_k=5):
        """Retrieves relevant chunks using FAISS search and hybrid re-ranking."""
        self.load_index() # Ensure index is loaded
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index not loaded or empty. Cannot retrieve.")
            return []
        if self.tfidf_matrix is None:
             logger.warning("TF-IDF matrix not fitted. Hybrid scoring may be inaccurate.")
             # Consider fitting it here if not already done, or default to pure semantic

        # 1. Encode Query
        # Use run_in_executor for potentially CPU/GPU intensive encoding
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda:self.model.encode([f"query: {query}"], # Add query prefix
            normalize_embeddings=True)
            
        )
        query_vec = query_embedding[0].astype(np.float32).reshape(1, -1)

        # 2. FAISS Search
        k_search = min(self.faiss_candidates, self.index.ntotal) # Don't request more candidates than exist
        try:
            distances, indices = self.index.search(query_vec, k_search)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

        faiss_ids = [int(i) for i in indices[0] if i != -1] # Get valid indices
        if not faiss_ids:
            return []

        # 3. Get Embeddings & Metadata for candidates
        candidate_vectors = np.array([self.index.reconstruct(idx) for idx in faiss_ids]) # Get vectors from index
        candidate_metadata = await get_metadata_by_faiss_ids(self.db_path, faiss_ids)
        if not candidate_metadata:
             logger.warning(f"Could not retrieve metadata for FAISS IDs: {faiss_ids}")
             return []

        # Create mapping for easy lookup
        metadata_map = {item['faiss_id']: item for item in candidate_metadata}
        vector_map = {faiss_id: vec for faiss_id, vec in zip(faiss_ids, candidate_vectors)}

        # 4. Calculate Hybrid Scores & Build Chunks
        results = []
        for idx, faiss_id in enumerate(faiss_ids):
            if faiss_id not in metadata_map or faiss_id not in vector_map:
                continue # Skip if metadata or vector is missing

            target_metadata = metadata_map[faiss_id]
            target_vector = vector_map[faiss_id]

            # Calculate hybrid score using the *original sentence vector*
            hybrid_s = self._hybrid_score(query_vec[0], target_vector, faiss_id)

            # Build contextual chunk around this central sentence
            # Need metadata for neighbors - fetch a slightly wider range around the candidate IDs
            min_id, max_id = min(faiss_ids), max(faiss_ids)
            context_fetch_ids = list(range(max(0, min_id - self.window_size), max_id + self.window_size + 1))
            # Efficiently fetch metadata needed for context building (can be optimized further)
            context_metadata = await get_metadata_by_faiss_ids(self.db_path, context_fetch_ids)
            if not context_metadata:
                 logger.warning(f"Could not retrieve context metadata around faiss_id {faiss_id}")
                 context_text = target_metadata['sentence'] # Fallback to single sentence
            else:
                 # Build the specific chunk for the current faiss_id
                 chunk_list = build_contextual_chunks(context_metadata, [faiss_id], self.window_size)
                 if chunk_list:
                      context_text = chunk_list[0]['text']
                 else:
                      context_text = target_metadata['sentence'] # Fallback


            results.append({
                "score": hybrid_s,
                "text": context_text, # Use the generated contextual chunk text
                "source": target_metadata["file"],
                "sentence_hash": target_metadata["sentence_hash"],
                "faiss_id": faiss_id
            })

        # 5. Sort and Deduplicate
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate based on sentence_hash (preferring higher score)
        seen_hashes = set()
        final_results = []
        for res in results:
            if res["sentence_hash"] not in seen_hashes:
                final_results.append(res)
                seen_hashes.add(res["sentence_hash"])
            if len(final_results) >= top_k:
                break

        return final_results


class KeywordSearcher:
    """Handles TF-IDF based keyword search."""
    def __init__(self, db_path):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=1)
        self.tfidf_matrix = None
        self.metadata = [] # Stores {faiss_id, sentence_hash, file, sentence}

    async def fit(self):
        """Loads data and fits the TF-IDF vectorizer."""
        logger.info("Fitting TF-IDF for KeywordSearcher...")
        self.metadata = await get_all_metadata_for_tfidf(self.db_path)
        if not self.metadata:
             logger.warning("No metadata found for KeywordSearcher TF-IDF fitting.")
             return
        sentences = [m['sentence'] for m in self.metadata]
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(sentences)
            logger.info(f"KeywordSearcher TF-IDF fitted on {len(self.metadata)} sentences.")
        except ValueError as e:
             logger.error(f"KeywordSearcher TF-IDF fitting error: {e}")
             self.tfidf_matrix = None


    def search(self, query, top_k=10):
        """
        Search using TF-IDF cosine similarity.
        Ensures results have a non-empty 'text' field.
        """
        # Input validation checks
        if self.tfidf_matrix is None:
             logger.warning("KeywordSearcher TF-IDF matrix is None. Cannot search.")
             return []
        if not hasattr(self, 'metadata') or not self.metadata:
             logger.warning("KeywordSearcher has no metadata. Cannot search.")
             return []
        if self.vectorizer is None:
             logger.warning("KeywordSearcher vectorizer is not initialized. Cannot search.")
             return []


        try:
            # Transform query and calculate similarities
            query_vec = self.vectorizer.transform([query])
            # Handle empty query vector case
            if query_vec.nnz == 0:
                 logger.warning(f"Query '{query}' resulted in an empty TF-IDF vector. No keyword results.")
                 return []

            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            num_docs = len(cosine_similarities)
            actual_top_k = min(top_k, num_docs) # Adjust top_k if less docs than requested

            # Get top-k results efficiently
            if num_docs == 0:
                 return [] # No documents to search
            elif num_docs > actual_top_k * 10 and actual_top_k > 0: # Use argpartition for large N
                # Get indices of the top k elements (unordered)
                top_indices_unsorted = np.argpartition(cosine_similarities, -actual_top_k)[-actual_top_k:]
                # Sort only the top k elements by score
                sorted_indices = top_indices_unsorted[np.argsort(cosine_similarities[top_indices_unsorted])][::-1]
            elif actual_top_k > 0: # Use argsort for smaller N or when k is close to N
                sorted_indices = np.argsort(cosine_similarities)[::-1][:actual_top_k]
            else: # case where actual_top_k is 0
                sorted_indices = []


            results = []
            for i in sorted_indices:
                score = cosine_similarities[i]
                # Add a small threshold to filter irrelevant results
                if score > 0.01:
                    # Check index bounds for safety
                    if i < len(self.metadata):
                        item = self.metadata[i]

                        # --- Solution 3 Implementation ---
                        # 1. Safely get the sentence text using .get()
                        sentence_text = item.get('sentence', '')

                        # 2. Check if the text is empty or None
                        if not sentence_text:
                            logger.warning(f"Empty sentence text found for index {i} (faiss_id {item.get('faiss_id', 'N/A')}, hash {item.get('sentence_hash', 'N/A')}). Skipping keyword result.")
                            continue # 3. Skip this result if text is empty

                        # --- End Solution 3 Implementation ---

                        # If text is valid, proceed to create the result dictionary
                        # Use .get() for other fields for robustness
                        results.append({
                            'text': item['sentence_text'], # Use the validated, non-empty text
                            'source': item.get('file', 'Unknown'),
                            'sentence_hash': item.get('sentence_hash', 'N/A'),
                            'faiss_id': item.get('faiss_id', -1),
                            'score': float(score) # Ensure score is standard float
                        })
                    else:
                        # This case should be rare if indices are generated correctly
                        logger.warning(f"Index {i} out of bounds for metadata during keyword search (similarity: {score:.4f}). Skipping.")

            return results
        except ValueError as ve:
             # Catch specific error if query term is not in vocabulary after fitting
             if "empty vocabulary" in str(ve) or "not fitted" in str(ve):
                 logger.warning(f"Keyword search failed for query '{query}'. It might contain only terms not in the vocabulary or vectorizer not fitted. {ve}")
             else:
                 logger.error(f"Keyword search failed with ValueError: {ve}", exc_info=True)
             return []
        except Exception as e:
            logger.error(f"Keyword search failed unexpectedly: {e}", exc_info=True)
            return []

# -----------------------------
# Google LLM Generation (Async)
# -----------------------------
async def google_text_generation(session, prompt, api_key, endpoint, temperature=0.2, max_output_tokens=256, top_p=0.95):
    """Generates text using Google's Generative AI API asynchronously."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "topP": top_p
            # Add other parameters like stop sequences if needed
        }
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    url = f"{endpoint}?key={api_key}" # Include API key in URL params

    try:
        # Use the provided aiohttp session
        async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response: # Increased timeout
            response.raise_for_status()
            result = await response.json()

            # Handle potential API response variations and errors
            if "candidates" in result and result["candidates"]:
                 candidate = result["candidates"][0]
                 if "content" in candidate and "parts" in candidate["content"] and candidate["content"]["parts"]:
                      return candidate["content"]["parts"][0].get("text", "")
                 elif "finishReason" in candidate:
                      logger.warning(f"LLM generation stopped. Reason: {candidate['finishReason']}. Details: {candidate.get('safetyRatings')}")
                      return f"[LLM Generation Stopped: {candidate['finishReason']}]"
                 else:
                      logger.error(f"Unexpected LLM candidate structure: {candidate}")
                      return "[LLM Response Error: Unexpected candidate structure]"
            elif "promptFeedback" in result:
                 logger.error(f"LLM prompt feedback error: {result['promptFeedback']}")
                 return f"[LLM Prompt Error: {result['promptFeedback'].get('blockReason', 'Unknown')}]"
            else:
                 logger.error(f"Unexpected LLM response structure: {result}")
                 return "[LLM Response Error: Unexpected structure]"

    except aiohttp.ClientResponseError as e:
        logger.error(f"❌ HTTP Error during Google LLM call: {e.status} {e.message}")
        # Attempt to read error body
        try:
            error_body = await response.text()
            logger.error(f"LLM Error Body: {error_body[:500]}") # Log first 500 chars
        except Exception as read_e:
            logger.error(f"Could not read error body: {read_e}")
        return f"[LLM HTTP Error: {e.status}]"
    except asyncio.TimeoutError:
         logger.error("❌ Timeout during Google LLM call.")
         return "[LLM Timeout Error]"
    except Exception as e:
        logger.error(f"❌ Unexpected error during Google LLM text generation: {e} (Type: {type(e)})")
        return "[LLM Unexpected Error]"


async def rag_generation(session, query, context_chunks, api_key, endpoint):
    """Generates a RAG response based on query and context chunks."""
    if not context_chunks:
        return "No relevant context found to generate a response."

    # Group chunks by source for better prompting
    context_by_source = defaultdict(list)
    for i, chunk in enumerate(context_chunks):
        context_by_source[chunk['source']].append(f"Excerpt {i+1}: {chunk['text']}")

    context_str = ""
    source_map = {}
    current_source_idx = 1
    for source, excerpts in context_by_source.items():
        context_str += f"\n**From {source} [Source {current_source_idx}]:**\n"
        context_str += "\n".join(excerpts) + "\n"
        source_map[current_source_idx] = source
        current_source_idx += 1


    prompt = f"""**Analyze the following resume excerpts based ONLY on the text provided, in relation to the Job Requirement.**

**Job Requirement:**
{query}

**Relevant Resume Excerpts:**
{context_str}

**Task:**
1.  **Overall Assessment:** Briefly state if the candidate appears to meet the core requirements based *only* on these excerpts.
2.  **Evidence:** For each key aspect of the job requirement met, provide the *specific quote* from the excerpts that supports it. Cite the source using [Source #] corresponding to the file name above.
3.  **Gaps:** If the excerpts clearly indicate a required skill is missing, state that.
4.  **Conciseness:** Be factual and concise. Do not invent qualifications or infer skills not explicitly mentioned.

**Response:**"""

    # Use lower temperature for more factual RAG generation
    return await google_text_generation(session, prompt, api_key, endpoint, temperature=0.1, max_output_tokens=1024)


async def rag_generation_hybrid(session, query, chunk, api_key, endpoint):
    
    prompt = f"""**Evaluate this single resume excerpt against the job requirement.**

**Job Requirement:**
{query}

**Resume Excerpt (from {chunk['source']}):**
"{chunk['text']}"

**Tasks:**
1.  **Direct Relevance:** Does this specific excerpt demonstrate experience directly relevant to the requirement? (Answer Yes/No).
2.  **Matching Skills:** If Yes, list the *specific skills or experiences* mentioned in the excerpt that match the requirement. Quote directly if possible.
3.  **Confidence:** Briefly state your confidence (High/Medium/Low) that this excerpt *alone* indicates a good match for the requirement.

**Response:**"""

    # Use lower temperature for more factual evaluation
    return await google_text_generation(session, prompt, api_key, endpoint, temperature=0.1, max_output_tokens=512)


async def score_relevance_with_llm(session, query, candidates, api_key, endpoint, db_path):
    """Re-ranks candidates using LLM scoring, incorporating caching and feedback."""
    feedback_data = load_feedback() # Keep feedback loading simple for now
    reranked_results = []

    # Prepare tasks for concurrent LLM scoring
    tasks = []
    candidate_map = {} # To map task results back to candidates

    for i, item in enumerate(candidates):
        if "text" not in item or "sentence_hash" not in item:
            logger.warning(f"⚠️ Skipping candidate missing required keys: {item}")
            continue
        text = item["text"]
        cache_key = make_cache_key(query, item['sentence_hash'])
        candidate_map[cache_key] = item # Store item for later retrieval

        # Check cache first (synchronously for simplicity here, could be async)
        cached = await get_from_cache(db_path, cache_key)
        if cached:
            logger.debug(f"Cache hit for: {query} | {item['sentence_hash'][:10]}...")
            item["llm_score"] = cached["score"]
            item["llm_explanation"] = cached["explanation"]
            # Apply feedback adjustment even if cached
            adjusted_score = apply_feedback(item["llm_score"], feedback_data, query, item["sentence_hash"])
            item["final_score"] = adjusted_score
            reranked_results.append(item)
        else:
            logger.debug(f"Cache miss for: {query} | {item['sentence_hash'][:10]}... Calling LLM.")
            # Prepare LLM call task
            prompt = f"""Assess the relevance of the following resume excerpt for the query. Score the match from 0.0 (no match) to 1.0 (strong match). Respond ONLY with the numeric score (float).

Query: "{query}"

Excerpt from {item['source']}:
"{text}"

Score (0.0-1.0):
"""
            # Create task: tuple of (cache_key, coro)
            tasks.append((cache_key, google_text_generation(session, prompt, api_key, endpoint, temperature=0.0, max_output_tokens=10))) # Temp 0 for scoring

    # Run LLM scoring tasks concurrently
    if tasks:
        logger.info(f"Running {len(tasks)} LLM scoring tasks concurrently...")
        # Use tqdm.gather for progress bar with asyncio.gather
        results = await tqdm.gather(*(task for _, task in tasks), desc="🧠 LLM Re-ranking")
        logger.info("LLM scoring tasks complete.")

        # Process results
        for i, score_text in enumerate(results):
            cache_key, _ = tasks[i] # Get corresponding cache key
            item = candidate_map[cache_key] # Get original candidate item

            try:
                llm_score = float(score_text.strip())
                llm_score = max(0.0, min(llm_score, 1.0)) # Clamp score
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse LLM score '{score_text}'. Defaulting to 0.0. Error: {e}")
                llm_score = 0.0

            explanation = score_text.strip() # Store the raw response as explanation

            # Store to cache asynchronously
            await store_to_cache(db_path, cache_key, llm_score, explanation)

            item["llm_score"] = llm_score
            item["llm_explanation"] = explanation

            # Apply feedback adjustment
            adjusted_score = apply_feedback(llm_score, feedback_data, query, item["sentence_hash"])
            item["final_score"] = adjusted_score
            reranked_results.append(item)

    # Sort final results by the feedback-adjusted score
    return sorted(reranked_results, key=lambda x: x.get("final_score", 0.0), reverse=True)


# -----------------------------
# Hybrid Search & Feedback Logic
# -----------------------------
LLM_SCORE_CACHE = {} # Keep in-memory cache as fallback/simplification if DB fails? Or remove.
# Using DB cache primarily now.

def make_cache_key(query, identifier):
    """Creates a unique cache key."""
    key_str = f"{query.lower().strip()}::{identifier}"
    return hashlib.sha256(key_str.encode()).hexdigest()

def normalize_scores(results, key="score"):
    """Normalizes scores in a list of result dicts to the 0-1 range."""
    if not results:
        return []
    scores = [r.get(key, 0.0) for r in results] # Default to 0.0 if key missing
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return [1.0] * len(results) # Avoid division by zero; assign max score
    return [(s - min_score) / (max_score - min_score) for s in scores]

def reciprocal_rank_fusion(rank, k=RRF_K):
    """Calculates RRF score."""
    return 1.0 / (k + rank)

def save_feedback(query, sentence_hash, is_relevant, feedback_file=DEFAULT_FEEDBACK_PATH):
    """Store user feedback with timestamp."""
    feedback_entry = {
        "query": query,
        "sentence_hash": sentence_hash,
        "is_relevant": is_relevant,
        "timestamp": datetime.now().isoformat()
    }
    try:
        # Append mode, create file if not exists
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry) + "\n")
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")

def load_feedback(feedback_file=DEFAULT_FEEDBACK_PATH):
    """Load all historical feedback."""
    feedback = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        feedback.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in feedback file: {line.strip()}")
                        continue
        except Exception as e:
            logger.error(f"Error loading feedback file {feedback_file}: {e}")
    return feedback

def apply_feedback(current_score, feedback_data, query, sentence_hash):
     """Adjusts score based on historical feedback."""
     # Simple adjustment - can be made more sophisticated
     adjustment = 0.0
     feedback_count = 0
     query_lower = query.lower()

     for f in feedback_data:
         # Match on sentence hash and case-insensitive query
         if f["sentence_hash"] == sentence_hash and f["query"].lower() == query_lower:
             adjustment += 0.1 if f["is_relevant"] else -0.15 # Stronger negative feedback
             feedback_count += 1

     if feedback_count > 0:
         logger.debug(f"Applying feedback adjustment: {adjustment} for {sentence_hash} on query '{query}'")
         # Apply adjustment but keep score within [0, 1]
         return max(0.0, min(1.0, current_score + adjustment))
     else:
         return current_score


async def hybrid_search(session, query, retriever, keyword_searcher, api_key, endpoint, db_path, top_k=10, rerank_with_llm=True):
    """Performs hybrid search combining vector and keyword results, with optional LLM re-ranking."""

    # 1. Retrieve from both sources concurrently
    logger.info("Performing hybrid retrieval...")
    vector_task = asyncio.create_task(retriever.retrieve(query, top_k=top_k * 2)) # Fetch more candidates initially
    # Keyword search is synchronous currently, run in executor if it becomes slow
    loop = asyncio.get_running_loop()
    keyword_results = await loop.run_in_executor(None, keyword_searcher.search, query, top_k * 2)
    vector_results = await vector_task
    logger.info(f"Retrieved {len(vector_results)} vector results and {len(keyword_results)} keyword results.")


    # 2. Normalize scores (handle cases with no results)
    vec_scores_norm = normalize_scores(vector_results, key="score") if vector_results else []
    kwd_scores_norm = normalize_scores(keyword_results, key="score") if keyword_results else []

    # 3. Combine using Reciprocal Rank Fusion (RRF)
    combined = defaultdict(lambda: {'score': 0.0, 'sources': set()})

    # Process vector results
    for rank, (item, norm_score) in enumerate(zip(vector_results, vec_scores_norm)):
        key = item['sentence_hash']
        combined[key]['score'] += reciprocal_rank_fusion(rank)
        combined[key].update({k: v for k, v in item.items() if k != 'score'}) # Update metadata, keep fused score
        combined[key]['sources'].add('vector')


    # Process keyword results
    for rank, (item, norm_score) in enumerate(zip(keyword_results, kwd_scores_norm)):
        key = item['sentence_hash']
        combined[key]['score'] += reciprocal_rank_fusion(rank)
        if key not in combined: # Add metadata if not seen in vector results
             combined[key].update({k: v for k, v in item.items() if k != 'score'})
        combined[key]['sources'].add('keyword')


    # 4. Sort fused results and take top_k
    # Convert sources set to list for JSON serialization if needed later
    for key in combined:
        combined[key]['sources'] = list(combined[key]['sources'])

    fused_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)[:top_k * 2] # Keep more for re-ranking
    logger.info(f"Fused results count: {len(fused_results)}")


    # 5. LLM Re-ranking (Optional)
    if rerank_with_llm and api_key and endpoint:
        logger.info("Starting LLM re-ranking...")
        # Ensure session is passed correctly
        reranked_candidates = await score_relevance_with_llm(session, query, fused_results, api_key, endpoint, db_path)
        logger.info(f"LLM re-ranking complete. Candidates count: {len(reranked_candidates)}")

        # Generate justifications concurrently for the final top_k
        final_results = reranked_candidates[:top_k]
        justification_tasks = [
            rag_generation_hybrid(session, query, item, api_key, endpoint)
            for item in final_results
        ]
        if justification_tasks:
             logger.info(f"Generating {len(justification_tasks)} justifications...")
             justifications = await tqdm.gather(*justification_tasks, desc="📝 Generating Justifications")
             logger.info("Justification generation complete.")
             for item, justification in zip(final_results, justifications):
                 item['justification'] = justification
        return final_results
    else:
        logger.info("Skipping LLM re-ranking.")
        # Return fused results directly if not re-ranking
        return fused_results[:top_k]


def display_results_with_feedback(results, query):
    """Interactive result display with feedback collection."""
    if not results:
        print("\n❌ No results found.")
        return

    print("\n--- Search Results ---")
    for idx, res in enumerate(results, 1):
        print(f"\n🏆 Result {idx} (ID: {res['faiss_id']})")
        print(f"   Source: {res['source']}")
        # Display score based on what's available (final_score if reranked, score otherwise)
        score_key = "final_score" if "final_score" in res else "score"
        print(f"   Score: {res.get(score_key, 'N/A'):.4f}")
        if "llm_score" in res:
            print(f"   LLM Score: {res['llm_score']:.4f} ({res.get('llm_explanation', 'N/A')})")
        if "sources" in res:
             print(f"   Found via: {', '.join(res['sources'])}")

        print(f"\n   Excerpt:\n   ...\n   {res['text']}\n   ...")

        if "justification" in res:
            print(f"\n   💡 Justification:\n   {res['justification']}")

        # Collect feedback
        while True:
            feedback = input("\n   Is this result relevant? (y/n/skip): ").lower().strip()
            if feedback in ("y", "n", "skip"):
                break
            print("   Invalid input. Please use y/n/skip")

        if feedback != "skip":
            save_feedback(
                query=query,
                sentence_hash=res["sentence_hash"],
                is_relevant=(feedback == "y")
            )
            print(f"   Feedback recorded ({'Relevant' if feedback == 'y' else 'Not Relevant'}).")
        else:
            print("   Result skipped.")
        print("-" * 20)



# -----------------------------
# Pydantic Models
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    rerank: bool = True

class SearchResult(BaseModel):
    text: str
    source: str
    score: float
    explanation: Optional[str] = None
    justification: Optional[str] = None
    sentence_hash: Optional[str] = None  # Add missing optional fields
    faiss_id: Optional[int] = None
    sources: Optional[List[str]] = None

class FeedbackRequest(BaseModel):
    query: str
    sentence_hash: str
    is_relevant: bool

class SystemStatus(BaseModel):
    status: str
    version: str = "1.0.0"
    index_count: int
    db_path: str
    model: str

# -----------------------------
# FastAPI Lifespan Management
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Initialize global components
    global model, retriever, keyword_searcher, aiohttp_session
    
    logger.info("Initializing application components...")
    
    # Initialize database
    await initialize_database(DEFAULT_DB_PATH)
    
    # Load sentence transformer model
    model = SentenceTransformer(DEFAULT_MODEL)
    
    # Initialize search components
    retriever = RAGRetriever(DEFAULT_INDEX_PATH, DEFAULT_DB_PATH, model)
    keyword_searcher = KeywordSearcher(DEFAULT_DB_PATH)
    
    # Fit TF-IDF models
    await asyncio.gather(
        retriever.fit_tfidf(),
        keyword_searcher.fit()
    )
    
    # Create aiohttp session
    aiohttp_session = aiohttp.ClientSession()
    
    yield  # Application is running
    
    # Cleanup resources
    logger.info("Cleaning up resources...")
    if aiohttp_session:
        await aiohttp_session.close()

# Initialize FastAPI app
app = FastAPI(
    title="Resume Search API",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/search", response_model=List[SearchResult])
async def api_search(
    request: SearchRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Perform hybrid search across resumes"""
    try:
        # Get API key from headers, request, or environment
        api_key = x_api_key or request.api_key or os.environ.get("GOOGLE_API_KEY")
        
        if request.rerank and not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="API key is required for LLM reranking"
            )

        results = await hybrid_search(
            session=aiohttp_session,
            query=request.query,
            retriever=retriever,
            keyword_searcher=keyword_searcher,
            api_key=api_key,
            endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent",
            db_path=DEFAULT_DB_PATH,
            top_k=request.top_k,
            rerank_with_llm=request.rerank
        )
        return results
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search operation failed"
        )

@app.post("/feedback")
async def api_feedback(feedback: FeedbackRequest):
    """Submit relevance feedback for search results"""
    try:
        save_feedback(
            query=feedback.query,
            sentence_hash=feedback.sentence_hash,
            is_relevant=feedback.is_relevant
        )
        return {"status": "feedback recorded"}
    
    except Exception as e:
        logger.error(f"Feedback failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feedback recording failed"
        )

@app.get("/status", response_model=SystemStatus)
async def api_status():
    """Get system status and statistics"""
    return {
        "status": "OK",
        "index_count": retriever.index.ntotal if retriever and retriever.index else 0,
        "db_path": DEFAULT_DB_PATH,
        "model": DEFAULT_MODEL
    }


# -----------------------------
# Main Asynchronous Function
# -----------------------------
async def main():
    parser_cli = argparse.ArgumentParser(
        description="Optimized RAG/Hybrid Resume Search Engine (Async + SQLite)"
    )
    # Paths and URLs
    parser_cli.add_argument("--pdf_dir", type=str, default="resumes", help="Directory containing resume files")
    parser_cli.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH, help="Path to the SQLite database file")
    parser_cli.add_argument("--index_path", type=str, default=DEFAULT_INDEX_PATH, help="Path to the FAISS index file")
    parser_cli.add_argument("--feedback_path", type=str, default=DEFAULT_FEEDBACK_PATH, help="Path to the feedback log file")
    parser_cli.add_argument("--tika_url", type=str, default=DEFAULT_TIKA_URL, help="Tika server endpoint URL")

    # Model and API
    parser_cli.add_argument("--model_name", type=str, default=DEFAULT_MODEL, help="Sentence Transformer model name")
    parser_cli.add_argument("--google_api_key", type=str, default=os.environ.get("GOOGLE_API_KEY"), help="Google API key for text generation (or set GOOGLE_API_KEY env var)")
    parser_cli.add_argument("--google_endpoint", type=str, default="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent", help="Google API endpoint URL (ensure model matches)") # Updated to v1beta and latest

    # Control Flags
    parser_cli.add_argument("--rebuild", action="store_true", help="Rebuild the FAISS index and clear relevant DB tables from scratch")
    parser_cli.add_argument("--skip_indexing", action="store_true", help="Skip the indexing phase (only run search)")
    parser_cli.add_argument("--no_rerank", action="store_true", help="Disable LLM re-ranking in hybrid search")

    # Search Parameters
    parser_cli.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of final results to display")
    parser_cli.add_argument("--faiss_candidates", type=int, default=DEFAULT_FAISS_CANDIDATES, help="Number of initial candidates to retrieve from FAISS")

    args = parser_cli.parse_args()

    # --- Database Initialization ---
    await initialize_database(args.db_path)

    # --- Rebuild Logic ---
    if args.rebuild:
        logger.warning("🚨 Rebuild selected! Removing existing index and clearing metadata/cache tables.")
        if os.path.exists(args.index_path):
            os.remove(args.index_path)
            logger.info(f"Removed existing index file: {args.index_path}")
        try:
            async with aiosqlite.connect(args.db_path) as db:
                await db.execute("DELETE FROM metadata")
                await db.execute("DELETE FROM llm_cache")
                # Optionally VACUUM to reclaim space
                # await db.execute("VACUUM")
                await db.commit()
            logger.info("Cleared metadata and llm_cache tables in the database.")
        except Exception as e:
            logger.error(f"Error clearing database tables during rebuild: {e}")
            # Decide whether to proceed or exit if DB clearing fails

    # --- Load Model ---
    logger.info(f"Loading Sentence Transformer model: {args.model_name}")
    try:
        model = SentenceTransformer(args.model_name)
    except Exception as e:
        logger.error(f"Fatal: Failed to load Sentence Transformer model: {e}")
        return # Cannot proceed without model

    # --- Indexing Phase ---
    if not args.skip_indexing:
        logger.info("🚀 Starting indexing phase...")
        start_time = time.time()
        new_metadata, new_vectors = await process_all_files(args.pdf_dir, model, args.tika_url, args.db_path)
        await build_incremental_index(new_metadata, new_vectors, model, args.index_path, args.db_path, args.rebuild)
        end_time = time.time()
        logger.info(f"⏳ Indexing phase completed in {end_time - start_time:.2f} seconds.")
    else:
        logger.info("⏩ Skipping indexing phase as requested.")


    # --- Prepare Search Components (Load data) ---
    logger.info("Initializing search components...")
    retriever = RAGRetriever(args.index_path, args.db_path, model, faiss_candidates=args.faiss_candidates)
    keyword_searcher = KeywordSearcher(args.db_path)

    # Fit TF-IDF models (can be done concurrently)
    fit_tasks = [retriever.fit_tfidf(), keyword_searcher.fit()]
    await asyncio.gather(*fit_tasks)
    logger.info("Search components initialized and TF-IDF fitted.")

    # --- Interactive Search Loop ---
    print("\n--- Resume Search Engine ---")
    print("Enter a job requirement or query.")
    print("Prefix with 'hybrid:' for hybrid search (requires API key for re-ranking/justification).")
    print("Type 'exit' to quit.")

    # Create a single aiohttp session for the duration of the search loop
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                query_input = input("\n🔎 Enter query (or 'exit'): ").strip()
                if query_input.lower() == "exit":
                    break
                if not query_input:
                    continue

                search_start_time = time.time()

                if query_input.lower().startswith("hybrid:"):
                    actual_query = query_input[len("hybrid:"):].strip()
                    if not actual_query:
                        print("Please provide a query after 'hybrid:'.")
                        continue

                    print(f"\n⚡ Performing Hybrid Search for: '{actual_query}'")
                    if not args.google_api_key:
                         logger.warning("Google API key not provided (--google_api_key or GOOGLE_API_KEY env var). LLM re-ranking/justification disabled.")
                         rerank_flag = False
                         api_key_to_use = None
                         endpoint_to_use = None
                    else:
                         rerank_flag = not args.no_rerank
                         api_key_to_use = args.google_api_key
                         endpoint_to_use = args.google_endpoint


                    hybrid_results = await hybrid_search(
                        session=session,
                        query=actual_query,
                        retriever=retriever,
                        keyword_searcher=keyword_searcher,
                        api_key=api_key_to_use,
                        endpoint=endpoint_to_use,
                        db_path=args.db_path,
                        top_k=args.top_k,
                        rerank_with_llm=rerank_flag
                    )
                    display_results_with_feedback(hybrid_results, actual_query)

                else:
                    # Default to RAG-enhanced search (Vector search + LLM justification)
                    actual_query = query_input
                    print(f"\n🤖 Performing RAG Search for: '{actual_query}'")

                    if not args.google_api_key:
                        print("❌ Google API key not provided. Cannot generate RAG justification.")
                        # Optionally show only retrieved chunks without justification
                        retrieved_chunks = await retriever.retrieve(actual_query, top_k=args.top_k)
                        display_results_with_feedback(retrieved_chunks, actual_query) # Display raw chunks
                        continue

                    # 1. Retrieve chunks
                    context_chunks = await retriever.retrieve(actual_query, top_k=args.top_k)

                    if not context_chunks:
                        print("No relevant qualifications found in resumes.")
                        continue

                    # 2. Group by resume and generate justification
                    # This part could also be parallelized if generating per resume
                    resume_contexts = defaultdict(list)
                    for chunk in context_chunks:
                         resume_contexts[chunk['source']].append(chunk)

                    print("\n--- RAG Results ---")
                    rag_results = []
                    for resume, chunks in resume_contexts.items():
                         print(f"\nAnalysing {resume}...")
                         justification = await rag_generation(session, actual_query, chunks, args.google_api_key, args.google_endpoint)
                         print(f"💡 Justification for {resume}:\n{justification}")
                         # Store results if needed later
                         rag_results.append({
                             "resume": resume,
                             "match_score": len(chunks), # Simple score based on chunk count
                             "justification": justification,
                             "source_chunks": chunks
                         })
                    # Optionally display results again in a structured way after all justifications

                search_end_time = time.time()
                print(f"\n⏳ Search completed in {search_end_time - search_start_time:.2f} seconds.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.exception("An error occurred during the search loop:") # Log full traceback
                print(f"\n❌ An unexpected error occurred: {e}. Please check logs.")
                # Optionally break or continue loop after error

    logger.info("Search session ended.")


if __name__ == "__main__":
    # Run the main asynchronous function
    parser = argparse.ArgumentParser(description="Resume Search Engine")
    parser.add_argument("--api", action="store_true", help="Run as an API server")
    args = parser.parse_args()

    if args.api:
        import uvicorn
        uvicorn.run(app, host=API_HOST, port=API_PORT)
    else:
        asyncio.run(main())
    