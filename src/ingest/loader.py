"""Load RAG chunks into Supabase university_knowledge table."""

import time

import config
from src.ingest.embeddings import embed_texts
from src.ingest.supabase_client import (
    clear_knowledge_table,
    get_existing_chunk_ids,
    get_supabase_client,
    insert_knowledge_rows,
)
from src.rag.processor import load_documents_jsonl
from src.utils.logger import setup_logger

log = setup_logger("ingest")


def _doc_to_row(doc: dict, embedding: list[float]) -> dict:
    meta = doc.get("metadata", {})
    return {
        "content": doc["text"],
        "embedding": embedding,
        "meta_data": {
            "chunk_id": doc.get("id") or meta.get("chunk_id"),
            "source_url": meta.get("source_url"),
            "title": meta.get("title"),
            "slug": meta.get("slug"),
            "chunk_index": meta.get("chunk_index"),
            "total_chunks": meta.get("total_chunks"),
            "domain": meta.get("domain"),
            "has_fee_info": meta.get("has_fee_info", False),
            "keywords": meta.get("keywords", ""),
        },
    }


def ingest_to_supabase(*, clear_existing: bool = True) -> int:
    documents = load_documents_jsonl()
    if not documents:
        log.warning("No documents to ingest")
        return 0

    client = get_supabase_client()

    if clear_existing:
        log.info("Clearing existing rows in %s", config.KNOWLEDGE_TABLE)
        clear_knowledge_table(client)
        existing_ids: set[str] = set()
    else:
        existing_ids = get_existing_chunk_ids(client)
        documents = [d for d in documents if d.get("id") not in existing_ids]
        log.info("Resuming ingest | skipping %d existing chunks", len(existing_ids))
        if not documents:
            log.info("All chunks already ingested")
            return 0

    total = len(existing_ids)
    batch_size = config.INGEST_BATCH_SIZE

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        texts = [d["text"] for d in batch]
        log.info(
            "Embedding batch %d-%d of %d remaining",
            i + 1,
            i + len(batch),
            len(documents),
        )
        embeddings = embed_texts(texts)
        rows = [_doc_to_row(doc, emb) for doc, emb in zip(batch, embeddings)]
        insert_knowledge_rows(client, rows)
        total += len(rows)
        log.info("Inserted %d rows (total %d)", len(rows), total)
        if i + batch_size < len(documents):
            time.sleep(config.INGEST_BATCH_DELAY_SECONDS)

    log.info("Ingest complete | %d chunks in %s", total, config.KNOWLEDGE_TABLE)
    return total - len(existing_ids) if not clear_existing else total
