"""Crawl university site and build RAG documents."""

import asyncio

from src.crawl import deep_crawl
from src.rag import build_rag_documents
from src.utils.logger import setup_logger

log = setup_logger()


async def main():
    log.info("Iqra University pipeline: crawl + RAG")
    log.info("Target: %s", "https://www.iqrauni.edu.pk/")

    pages = await deep_crawl()
    if not pages:
        log.error("No pages crawled. Check output/crawl.log")
        return

    build_rag_documents(pages)
    log.info("Done. RAG output: output/rag/documents.jsonl")
    log.info("Next: python ingest.py")


if __name__ == "__main__":
    asyncio.run(main())
