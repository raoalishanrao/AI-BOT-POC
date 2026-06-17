"""Embed RAG chunks and load into Supabase."""

import argparse

from src.ingest import ingest_to_supabase
from src.utils.logger import setup_logger

log = setup_logger()


def main():
    parser = argparse.ArgumentParser(description="Ingest RAG documents into Supabase")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append without clearing university_knowledge table",
    )
    args = parser.parse_args()

    count = ingest_to_supabase(clear_existing=not args.keep_existing)
    log.info("Ingested %d knowledge chunks", count)


if __name__ == "__main__":
    main()
