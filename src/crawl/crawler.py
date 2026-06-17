"""Deep crawl Iqra University site with Crawl4AI for RAG."""

import asyncio
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

import config
from src.utils.logger import setup_logger

log = setup_logger()

SKIP_LINK_PARTS = ("/_files/", ".pdf", "admissions.iuck.org")

EXPAND_TABS_JS = """
(async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const clickables = [
    ...document.querySelectorAll('[role="tab"]'),
    ...document.querySelectorAll('.wixui-tabs__item'),
    ...document.querySelectorAll('button[aria-controls]'),
    ...document.querySelectorAll('[data-testid="stylablebutton-label"]'),
  ];
  for (const el of clickables) {
    try {
      el.scrollIntoView({ block: 'center' });
      el.click();
      await sleep(450);
    } catch (e) {}
  }
  const steps = 8;
  for (let i = 1; i <= steps; i++) {
    window.scrollTo(0, (document.body.scrollHeight * i) / steps);
    await sleep(400);
  }
  window.scrollTo(0, 0);
  await sleep(800);
})();
"""

INTERACTIVE_URL_PATTERNS = (
    "/fees",
    "/programs",
    "/admissions",
    "/faculty",
    "/department-",
    "/allied-health",
    "/business-administration",
    "/computing-technology",
    "/copy-of-pharmacy",
    "/nursing",
)

_NAV_START = re.compile(r"^top of page\s*$", re.M)
_FEE_STRUCTURE = re.compile(r"^#\s+Fee Structure", re.M)
_USE_TAB = re.compile(r"^Use tab to navigate through the menu items\.\s*$", re.M)
_APPLY_NOW = re.compile(r"^\[APPLY NOW\].*$", re.M)
_BOTTOM = re.compile(r"^bottom of page\s*$", re.M)
_SUBSCRIBE = re.compile(r"^##### Subscribe to Our website\s*$", re.M)


def _needs_interaction(url: str) -> bool:
    path = url.lower()
    return any(p in path for p in INTERACTIVE_URL_PATTERNS)


def _clean_markdown(text: str) -> str:
    text = text.strip()

    fee_match = _FEE_STRUCTURE.search(text)
    if fee_match:
        text = text[fee_match.start() :]
    else:
        for pattern in (r"^#\s+.+", r"^##\s+.+"):
            match = re.search(pattern, text, re.M)
            if match and match.start() > 200:
                text = text[match.start() :]
                break

    text = _NAV_START.sub("", text)
    text = _USE_TAB.sub("", text)
    text = _APPLY_NOW.sub("", text)
    text = _BOTTOM.sub("", text)
    text = _SUBSCRIBE.sub("", text)

    lines = text.splitlines()
    cleaned: list[str] = []
    in_nav = False
    for line in lines:
        stripped = line.strip()
        if not cleaned and (stripped.startswith("* [") or stripped.startswith("[![")):
            in_nav = True
            continue
        if in_nav and stripped.startswith("#"):
            in_nav = False
        if in_nav:
            continue
        if stripped == "[Apply](https://admissions.iuck.org/login)":
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\[!\[WhatsApp\].*$", "", text, flags=re.M | re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _slugify_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/") or "home"
    slug = re.sub(r"[^\w\-]", "_", path)
    return slug[:120]


def _build_run_config(url: str) -> CrawlerRunConfig:
    js_code = [EXPAND_TABS_JS] if _needs_interaction(url) else None
    delay = 4.0 if "/fees" in url.lower() else (3.0 if _needs_interaction(url) else 2.0)

    return CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(),
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=5,
        verbose=True,
        page_timeout=config.PAGE_TIMEOUT_SECONDS * 1000,
        wait_until="domcontentloaded",
        delay_before_return_html=delay,
        js_code=js_code,
        semaphore_count=1,
    )


def _extract_markdown(result) -> str:
    md = result.markdown
    if md is None:
        return ""
    if hasattr(md, "raw_markdown") and md.raw_markdown:
        raw = md.raw_markdown.strip()
    elif hasattr(md, "fit_markdown") and md.fit_markdown:
        raw = md.fit_markdown.strip()
    else:
        raw = str(md).strip()
    return _clean_markdown(raw)


def _normalize_url(url: str, base: str = "") -> str:
    joined = urljoin(base, url) if base else url
    clean, _ = urldefrag(joined)
    parsed = urlparse(clean)
    if parsed.scheme not in ("http", "https"):
        return ""
    scheme = "https"
    netloc = parsed.netloc.lower().removeprefix("www.")
    if netloc == "iqrauni.edu.pk":
        netloc = "www.iqrauni.edu.pk"
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def _is_allowed(url: str) -> bool:
    lower = url.lower()
    if any(part in lower for part in SKIP_LINK_PARTS):
        return False
    return config.ALLOWED_DOMAIN in urlparse(url).netloc.lower()


def _discover_links(result, source_url: str) -> list[str]:
    links = result.links.get("internal", []) if result.links else []
    found: list[str] = []
    seen: set[str] = set()
    for link in links:
        href = link.get("href") if isinstance(link, dict) else link
        if not href:
            continue
        normalized = _normalize_url(href, source_url)
        if normalized and _is_allowed(normalized) and normalized not in seen:
            seen.add(normalized)
            found.append(normalized)
    return found


def _result_to_page(result, url: str) -> dict | None:
    if not result.success:
        return None

    markdown = _extract_markdown(result)
    word_count = len(markdown.split())
    if len(markdown) < 30:
        return None

    title = result.metadata.get("title", "") if result.metadata else ""
    slug = _slugify_url(url)
    page_file = config.PAGES_DIR / f"{slug}.md"
    page_file.write_text(markdown, encoding="utf-8")

    return {
        "url": url,
        "title": title,
        "slug": slug,
        "markdown": markdown,
        "word_count": word_count,
        "file": str(page_file.relative_to(config.BASE_DIR)),
    }


def _save_manifest(pages: list[dict]) -> None:
    manifest = {
        "source": config.START_URL,
        "domain": config.ALLOWED_DOMAIN,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "max_depth": config.MAX_DEPTH,
        "max_pages": config.MAX_PAGES,
        "pages_crawled": len(pages),
        "pages": [
            {
                "url": p["url"],
                "title": p["title"],
                "word_count": p["word_count"],
                "file": p["file"],
            }
            for p in pages
        ],
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = config.OUTPUT_DIR / "crawl_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.debug("Manifest saved (%d pages)", len(pages))


async def _fetch_with_timeout(crawler, url: str, run_config: CrawlerRunConfig):
    started = time.monotonic()
    done = asyncio.Event()

    async def heartbeat():
        tick = 0
        while not done.is_set():
            await asyncio.sleep(config.HEARTBEAT_INTERVAL_SECONDS)
            if done.is_set():
                break
            tick += 1
            elapsed = time.monotonic() - started
            log.info(
                "  ... still fetching (%ds elapsed, heartbeat #%d) | %s",
                int(elapsed),
                tick,
                url,
            )

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        log.debug("Starting arun for %s (timeout=%ds)", url, config.PAGE_TIMEOUT_SECONDS)
        result = await asyncio.wait_for(
            crawler.arun(url=url, config=run_config, magic=True),
            timeout=config.PAGE_TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - started
        log.info("  fetch done in %.1fs", elapsed)
        return result
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - started
        log.warning("  TIMEOUT after %.1fs — skipping %s", elapsed, url)
        return None
    except Exception as exc:
        elapsed = time.monotonic() - started
        log.error("  ERROR after %.1fs — %s: %s", elapsed, url, exc)
        return None
    finally:
        done.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


async def deep_crawl() -> list[dict]:
    config.PAGES_DIR.mkdir(parents=True, exist_ok=True)

    browser_config = BrowserConfig(headless=True, verbose=True)

    pages: list[dict] = []
    visited: set[str] = set()
    start = _normalize_url(config.START_URL)
    queue: deque[tuple[str, int]] = deque([(start, 0)])

    log.info("=" * 60)
    log.info(
        "Crawl started | target=%s | max_pages=%d | max_depth=%d",
        start,
        config.MAX_PAGES,
        config.MAX_DEPTH,
    )
    log.info("Log file: %s", config.LOG_FILE)
    log.info("=" * 60)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        while queue and len(pages) < config.MAX_PAGES:
            url, depth = queue.popleft()

            if url in visited:
                log.debug("SKIP already visited: %s", url)
                continue
            visited.add(url)

            log.info(
                "PAGE %d/%d | depth=%d | queue=%d | visited=%d | %s",
                len(pages) + 1,
                config.MAX_PAGES,
                depth,
                len(queue),
                len(visited),
                url,
            )

            result = await _fetch_with_timeout(crawler, url, _build_run_config(url))
            if result is None:
                continue
            if not result.success:
                log.warning("  FAILED: %s", result.error_message or "unknown error")
                continue

            page = _result_to_page(result, url)
            if page is None:
                log.warning("  SKIP insufficient content for %s", url)
                continue

            pages.append(page)
            log.info(
                "  SAVED %s (%d words) -> %s",
                page["title"] or page["slug"],
                page["word_count"],
                Path(page["file"]).name,
            )

            if depth < config.MAX_DEPTH:
                new_links = _discover_links(result, url)
                added = 0
                for link in new_links:
                    if link not in visited and link not in {u for u, _ in queue}:
                        queue.append((link, depth + 1))
                        added += 1
                log.info(
                    "  LINKS found=%d | new_queued=%d | queue_now=%d",
                    len(new_links),
                    added,
                    len(queue),
                )
            else:
                log.debug("  Max depth reached, not queuing child links")

            _save_manifest(pages)
            await asyncio.sleep(0.3)

    _save_manifest(pages)
    log.info("=" * 60)
    log.info(
        "Crawl finished | pages=%d | visited=%d | remaining_queue=%d",
        len(pages),
        len(visited),
        len(queue),
    )
    log.info("Manifest: %s", config.OUTPUT_DIR / "crawl_manifest.json")
    log.info("=" * 60)
    return pages


if __name__ == "__main__":
    asyncio.run(deep_crawl())
