# Ingestion — `ingest.py`

Turns `enwiki-latest-pages-articles.xml.bz2` into `articles.jsonl`.

## What it does

Streaming parse of the MediaWiki XML dump (`xml.etree.ElementTree.iterparse`
over a `bz2` stream — never loads the whole dump into memory). For each
`<page>`, it:

1. Skips redirects (`<redirect>` element).
2. Skips non-main-namespace pages (`<ns> != 0`) — talk pages, categories, etc.
3. Skips articles with `< 100` chars of raw wikitext.
4. Cleans wikitext via `mwparserfromhell` (`clean_wikitext()`).
5. Skips if cleaned text is `< 50` chars (templates/tables that strip to nothing).
6. Writes `{title, text, url}` as one JSONL line.

`url` is derived from the title: `https://en.wikipedia.org/wiki/{Title_With_Underscores}`.
This is the URL surfaced in the UI's "Sources" expander.

## CLI

```bash
python3 ingest.py --max-articles 5000 \
                  --dump enwiki-latest-pages-articles.xml.bz2 \
                  --output articles.jsonl
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-articles` | `5000` | Stop after this many articles written |
| `--dump` | `enwiki-latest-pages-articles.xml.bz2` | Input dump path |
| `--output` | `articles.jsonl` | Output JSONL path |

The dump is **not** included (gitignored, ~20GB). Download from
https://dumps.wikimedia.org/enwiki/latest/.

## Output format

`articles.jsonl`, one JSON object per line:

```json
{"title": "Abraham Lincoln", "text": "Abraham Lincoln was the 16th ...", "url": "https://en.wikipedia.org/wiki/Abraham_Lincoln"}
```

This is the contract `index.py` reads: it expects `title`, `text`, `url` keys.

## wikitext cleaning

`clean_wikitext()` (`ingest.py:17`) uses `mwparserfromhell.parse(...).strip_code()`
to strip wiki markup, then collapses blank lines into single-space-joined text.
On parse failure it returns `''`, which downstream becomes a "clean_fail" skip.

## Skipping logic (in order)

| Reason | Check | Where |
|--------|-------|-------|
| `redirect` | `<redirect>` element present | `ingest.py:56` |
| `non_main` | `<ns>` missing or `!= "0"` | `ingest.py:61` |
| `short` | raw wikitext `< 100` chars | `ingest.py:67` |
| `clean_fail` | cleaned text `< 50` chars | `ingest.py:73` |

Counts are printed at the end. Useful for sanity-checking the dump.
