"""
src/database/duplicates.py
Two-tier duplicate document detection:
1. Deterministic identity matching via SHA-256 hashing.
2. Semantic vector similarity via pgvector / embedding cosine similarity.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from sqlalchemy.orm import Session

from schemas import DuplicateMatchType, DuplicateResult, ExtractedField
from src.database.postgis_ext import EMBEDDING_DIM, is_postgres
from src.utils.hashing import compute_structured_field_hash

# Deterministic, offline, no-model-download text embedding. Uses feature hashing
# (the "hashing trick") rather than a downloaded transformer model, so it works
# without internet access and produces a fixed EMBEDDING_DIM-length dense vector
# suitable for storage in a pgvector column and real cosine-distance queries.
_hashing_vectorizer = HashingVectorizer(
    n_features=EMBEDDING_DIM, ngram_range=(1, 2), norm="l2", alternate_sign=False
)


def embed_text(text_value: str) -> np.ndarray:
    """Computes a deterministic EMBEDDING_DIM-dim L2-normalized embedding for storage in pgvector."""
    if not text_value or not text_value.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec = _hashing_vectorizer.transform([text_value]).toarray()[0]
    return vec.astype(np.float32)


class DuplicateDetector:
    """
    Manages document indexing and checks for exact hash duplicates or near-duplicate semantic matches.
    """

    def __init__(self, vector_similarity_threshold: float = 0.85):
        self.similarity_threshold = vector_similarity_threshold
        # In-memory document registry for standalone testability
        self._hash_registry: Dict[str, str] = {}  # sha256 -> doc_id
        self._field_hash_registry: Dict[str, str] = {}  # structured_hash -> doc_id
        self._doc_texts: List[str] = []
        self._doc_ids: List[str] = []
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        self._tfidf_matrix = None

    def register_document(
        self,
        document_id: str,
        sha256_hash: str,
        full_text: str,
        fields: Optional[Dict[str, ExtractedField]] = None,
        db_session: Optional[Session] = None,
    ):
        """Indexes a processed document into the duplicate detection registry."""
        self._hash_registry[sha256_hash] = document_id
        
        if fields:
            raw_dict = {k: str(v.normalized_value) for k, v in fields.items()}
            f_hash = compute_structured_field_hash(raw_dict)
            self._field_hash_registry[f_hash] = document_id

        # Real pgvector path: upsert the document's embedding into Postgres.
        if db_session is not None and is_postgres(db_session.get_bind()) and full_text.strip():
            self._upsert_embedding_postgis(db_session, document_id, full_text)

        if full_text.strip():
            self._doc_texts.append(full_text)
            self._doc_ids.append(document_id)
            # Recompute TF-IDF matrix (SQLite / no-Postgres fallback registry).
            if len(self._doc_texts) >= 1:
                self._tfidf_matrix = self._vectorizer.fit_transform(self._doc_texts)

    def _upsert_embedding_postgis(self, db_session: Session, document_id: str, full_text: str) -> None:
        """Stores/updates this document's embedding in the pgvector document_embeddings table."""
        try:
            vec = embed_text(full_text)
            vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec.tolist()) + "]"
            db_session.execute(
                text(
                    "INSERT INTO document_embeddings (document_id, embedding) "
                    "VALUES (:doc_id, CAST(:embedding AS vector)) "
                    "ON CONFLICT (document_id) DO UPDATE SET embedding = EXCLUDED.embedding;"
                ),
                {"doc_id": document_id, "embedding": vec_literal},
            )
            db_session.commit()
        except Exception as e:
            logger.warning(f"[DUPLICATES] Failed to upsert pgvector embedding for '{document_id}': {e}")
            db_session.rollback()

    def _query_pgvector_similar(
        self, db_session: Session, document_id: str, full_text: str
    ) -> Optional[Tuple[str, float]]:
        """
        Real pgvector nearest-neighbor query using the <=> cosine-distance operator.
        Returns (matched_document_id, similarity_score) for the closest OTHER document
        above the similarity threshold, or None.
        """
        try:
            vec = embed_text(full_text)
            vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec.tolist()) + "]"
            row = db_session.execute(
                text(
                    "SELECT document_id, 1 - (embedding <=> CAST(:embedding AS vector)) AS similarity "
                    "FROM document_embeddings "
                    "WHERE document_id != :doc_id "
                    "ORDER BY embedding <=> CAST(:embedding AS vector) ASC "
                    "LIMIT 1;"
                ),
                {"embedding": vec_literal, "doc_id": document_id},
            ).mappings().first()
            if row is None:
                return None
            score = float(row["similarity"])
            if score >= self.similarity_threshold:
                return row["document_id"], score
            return None
        except Exception as e:
            logger.warning(f"[DUPLICATES] pgvector similarity query failed: {e}")
            return None

    def check_duplicate(
        self,
        document_id: str,
        sha256_hash: str,
        full_text: str,
        fields: Optional[Dict[str, ExtractedField]] = None,
        db_session: Optional[Session] = None,
    ) -> DuplicateResult:
        """
        Runs SHA-256 exact match check and vector similarity check against registered records.
        """
        # 1. Check Exact SHA-256 File Hash
        if sha256_hash in self._hash_registry:
            matched_id = self._hash_registry[sha256_hash]
            if matched_id != document_id:
                return DuplicateResult(
                    is_duplicate=True,
                    match_type=DuplicateMatchType.EXACT_HASH,
                    matched_document_id=matched_id,
                    sha256_match=True,
                    similarity_score=1.0,
                    potential_duplicate_ids=[matched_id],
                    evidence=f"Exact SHA-256 file hash match with document '{matched_id}'",
                )

        # 2. Check Structured Field Exact Match
        if fields:
            raw_dict = {k: str(v.normalized_value) for k, v in fields.items()}
            f_hash = compute_structured_field_hash(raw_dict)
            if f_hash in self._field_hash_registry:
                matched_id = self._field_hash_registry[f_hash]
                if matched_id != document_id:
                    return DuplicateResult(
                        is_duplicate=True,
                        match_type=DuplicateMatchType.FIELD_EXACT,
                        matched_document_id=matched_id,
                        sha256_match=False,
                        similarity_score=1.0,
                        potential_duplicate_ids=[matched_id],
                        evidence=f"Identical canonical structured fields match with document '{matched_id}'",
                    )

        # 3. Check Vector Semantic Similarity
        # Real pgvector cosine-distance query when connected to Postgres; falls back
        # to the in-memory TF-IDF cosine-similarity registry otherwise (SQLite dev).
        if db_session is not None and is_postgres(db_session.get_bind()) and full_text.strip():
            pg_match = self._query_pgvector_similar(db_session, document_id, full_text)
            if pg_match is not None:
                matched_id, score = pg_match
                return DuplicateResult(
                    is_duplicate=True,
                    match_type=DuplicateMatchType.VECTOR_SIMILARITY,
                    matched_document_id=matched_id,
                    sha256_match=False,
                    similarity_score=round(score, 4),
                    potential_duplicate_ids=[matched_id],
                    evidence=f"High pgvector cosine similarity ({score:.3f}) with document '{matched_id}' (real PostGIS/pgvector query)",
                )

        elif self._tfidf_matrix is not None and full_text.strip() and len(self._doc_texts) > 0:
            try:
                query_vec = self._vectorizer.transform([full_text])
                similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
                
                # Exclude self if already indexed
                top_indices = np.argsort(similarities)[::-1]
                for idx in top_indices:
                    matched_id = self._doc_ids[idx]
                    score = float(similarities[idx])
                    if matched_id != document_id and score >= self.similarity_threshold:
                        return DuplicateResult(
                            is_duplicate=True,
                            match_type=DuplicateMatchType.VECTOR_SIMILARITY,
                            matched_document_id=matched_id,
                            sha256_match=False,
                            similarity_score=round(score, 4),
                            potential_duplicate_ids=[matched_id],
                            evidence=f"High vector cosine similarity ({score:.3f}) with document '{matched_id}'",
                        )
            except Exception:
                pass

        return DuplicateResult(
            is_duplicate=False,
            match_type=DuplicateMatchType.NONE,
            similarity_score=0.0,
            potential_duplicate_ids=[],
        )
