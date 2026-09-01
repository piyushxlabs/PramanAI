"""Local Sovereign Multilingual Neural Cross-Encoder Reranker using FlashRank (ONNX).

Provides zero-overhead, CPU/GPU-friendly neural cross-attention reranking for
Uttarakhand Government Order passages retrieved from Layer 2 Hybrid Search,
supporting Devanagari Hindi, English, and Hinglish administrative texts.
"""

import logging
import os
from typing import Any, Optional
from src.tools.schemas.search_go_corpus import PassageMatch

logger = logging.getLogger("shasanai.reranker")

_ranker_instance: Optional[Any] = None
DEFAULT_CACHE_DIR = "data/models/reranker"


def get_reranker() -> Optional[Any]:
    """Returns singleton FlashRank Ranker instance with multilingual model support."""
    global _ranker_instance
    if _ranker_instance is None:
        try:
            from flashrank import Ranker

            os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

            # Attempt multilingual model initialization (bge-reranker-v2-m3 / ms-marco-MultiBERT-L-12)
            try:
                _ranker_instance = Ranker(
                    model_name="bge-reranker-v2-m3",
                    cache_dir=DEFAULT_CACHE_DIR,
                )
                logger.info("FlashRank multilingual cross-encoder (bge-reranker-v2-m3) initialized.")
            except Exception:
                logger.info("bge-reranker-v2-m3 not in repo; initializing FlashRank multilingual ms-marco-MultiBERT-L-12.")
                _ranker_instance = Ranker(
                    model_name="ms-marco-MultiBERT-L-12",
                    cache_dir=DEFAULT_CACHE_DIR,
                )
                logger.info("FlashRank multilingual cross-encoder (ms-marco-MultiBERT-L-12) initialized successfully.")
        except Exception as exc:
            logger.warning("FlashRank initialization warning (%s): %s; using passage score ranking", type(exc).__name__, exc)
            _ranker_instance = None
    return _ranker_instance


def rerank_passages(
    query_text: str,
    passages: list[PassageMatch],
    top_k: int = 5,
) -> list[PassageMatch]:
    """Reranks candidate passages against query_text using local Multilingual Neural Cross-Encoder."""
    if not passages or len(passages) <= 1:
        return passages

    ranker = get_reranker()
    if ranker is None:
        # Sort by existing relevance score
        sorted_passages = sorted(passages, key=lambda p: float(p.relevance_score), reverse=True)
        return sorted_passages[:top_k]

    try:
        from flashrank import RerankRequest

        passages_payload = [
            {
                "id": idx,
                "text": p.exact_text_excerpt[:1000],
                "meta": p,
            }
            for idx, p in enumerate(passages)
        ]

        rerank_req = RerankRequest(query=query_text, passages=passages_payload)
        results = ranker.rerank(rerank_req)

        reranked_passages: list[PassageMatch] = []
        for r in results[:top_k]:
            passage_obj: PassageMatch = r["meta"]
            raw_score = float(r.get("score", passage_obj.relevance_score))
            # Clamp to [0.0, 1.0]
            passage_obj.relevance_score = max(0.0, min(1.0, raw_score))
            reranked_passages.append(passage_obj)

        logger.info("Neural Cross-Encoder reranked %d -> %d passages", len(passages), len(reranked_passages))
        return reranked_passages

    except Exception as exc:
        logger.warning("FlashRank reranking failed (%s): %s; returning original passages", type(exc).__name__, exc)
        sorted_passages = sorted(passages, key=lambda p: float(p.relevance_score), reverse=True)
        return sorted_passages[:top_k]
