"""Build RAG documents from crawled pages."""

import json
from datetime import datetime, timezone
from pathlib import Path

import config
from src.rag.chunker import chunk_text
from src.rag.fees import SLUG_TO_PROGRAM, build_fee_lookup, enrich_page_markdown, parse_fees_markdown
from src.utils.logger import setup_logger

log = setup_logger("rag")


def load_pages_from_manifest() -> list[dict]:
    manifest_path = config.OUTPUT_DIR / "crawl_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("crawl_manifest.json not found. Run main.py first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = []
    for page_info in manifest["pages"]:
        md_path = config.BASE_DIR / page_info["file"]
        pages.append(
            {
                "url": page_info["url"],
                "title": page_info["title"],
                "slug": md_path.stem,
                "markdown": md_path.read_text(encoding="utf-8"),
            }
        )
    return pages


def load_documents_jsonl(path: Path | None = None) -> list[dict]:
    jsonl_path = path or (config.RAG_DIR / "documents.jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found. Run main.py first.")

    documents = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def _load_fees_from_pages(pages: list[dict]) -> list[dict]:
    for page in pages:
        if page["slug"] == "fees" or page["url"].rstrip("/").endswith("/fees"):
            return parse_fees_markdown(page["markdown"])
    fees_path = config.PAGES_DIR / "fees.md"
    if fees_path.exists():
        return parse_fees_markdown(fees_path.read_text(encoding="utf-8"))
    return []


def _keywords_from_metadata(meta: dict) -> str:
    parts = [
        meta.get("title", ""),
        meta.get("source_url", ""),
        meta.get("slug", ""),
        "fees tuition admission" if meta.get("has_fee_info") else "",
    ]
    return " ".join(p for p in parts if p)


def build_rag_documents(pages: list[dict]) -> list[dict]:
    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[dict] = []

    fee_records = _load_fees_from_pages(pages)
    fee_lookup = build_fee_lookup(fee_records)
    if fee_records:
        fees_index = config.RAG_DIR / "fees_index.json"
        fees_index.write_text(json.dumps(fee_records, indent=2, ensure_ascii=False), encoding="utf-8")

    enriched_count = 0
    unmatched_program_slugs: list[str] = []

    for page in pages:
        enriched = enrich_page_markdown(page, fee_lookup)
        if enriched != page["markdown"]:
            enriched_count += 1
        elif page["slug"] in SLUG_TO_PROGRAM:
            unmatched_program_slugs.append(page["slug"])

        chunks = chunk_text(enriched, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            has_fee = "Fee Structure (Iqra University" in chunk
            meta = {
                "chunk_id": f"{page['slug']}_chunk_{i}",
                "source_url": page["url"],
                "title": page["title"],
                "slug": page["slug"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "domain": config.ALLOWED_DOMAIN,
                "has_fee_info": has_fee,
            }
            meta["keywords"] = _keywords_from_metadata(meta)
            documents.append({"id": meta["chunk_id"], "text": chunk, "metadata": meta})

    jsonl_path = config.RAG_DIR / "documents.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": config.START_URL,
        "total_pages": len(pages),
        "total_chunks": len(documents),
        "pages_with_fee_enrichment": enriched_count,
        "fee_programs_parsed": len(fee_records),
        "unmatched_program_slugs": unmatched_program_slugs,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "output_file": str(jsonl_path.relative_to(config.BASE_DIR)),
    }
    (config.RAG_DIR / "rag_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    log.info("Created %d RAG chunks from %d pages", len(documents), len(pages))
    log.info("Fee programs parsed: %d | pages enriched: %d", len(fee_records), enriched_count)
    if unmatched_program_slugs:
        log.warning("Program slugs without fee match: %s", ", ".join(unmatched_program_slugs))
    log.info("Output: %s", jsonl_path)
    return documents
