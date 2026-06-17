"""Supabase client helpers."""

import config
from src.utils.logger import setup_logger

log = setup_logger("ingest")


def get_supabase_client():
    if not config.SUPABASE_URL:
        raise ValueError("SUPABASE_URL must be set in .env")

    api_key = config.SUPABASE_SERVICE_KEY or config.SUPABASE_ANON_KEY
    if not api_key:
        raise ValueError(
            "Set SUPABASE_SERVICE_KEY (recommended) or NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY in .env"
        )
    if not config.SUPABASE_SERVICE_KEY:
        log.warning("Using publishable/anon key — ensure RLS allows insert on university_knowledge")

    from supabase import create_client

    return create_client(config.SUPABASE_URL, api_key)


def get_existing_chunk_ids(client) -> set[str]:
    response = client.table(config.KNOWLEDGE_TABLE).select("meta_data").execute()
    ids: set[str] = set()
    for row in response.data or []:
        meta = row.get("meta_data") or {}
        chunk_id = meta.get("chunk_id")
        if chunk_id:
            ids.add(chunk_id)
    return ids


def clear_knowledge_table(client) -> None:
    client.table(config.KNOWLEDGE_TABLE).delete().neq("id", 0).execute()


def insert_knowledge_rows(client, rows: list[dict]) -> None:
    if not rows:
        return
    client.table(config.KNOWLEDGE_TABLE).insert(rows).execute()
