#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

import argparse
import bz2
import json
import sys
import xml.etree.ElementTree as ET

import mwparserfromhell
from tqdm import tqdm

NS = 'http://www.mediawiki.org/xml/export-0.11/'


def clean_wikitext(wikitext):
    try:
        parsed = mwparserfromhell.parse(wikitext)
        text = parsed.strip_code(normalize=False, collapse=False, keep_template_params=False)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return ' '.join(lines).strip()
    except Exception:
        return ''


def main():
    parser = argparse.ArgumentParser(description='Extract articles from Wikipedia dump')
    parser.add_argument('--max-articles', type=int, default=5000)
    parser.add_argument('--dump', default='enwiki-latest-pages-articles.xml.bz2')
    parser.add_argument('--output', default='articles.jsonl')
    args = parser.parse_args()

    f = bz2.BZ2File(args.dump)
    context = ET.iterparse(f, events=('end',))

    out_file = open(args.output, 'w')
    count = 0
    total_pages = 0
    skipped = {'redirect': 0, 'non_main': 0, 'short': 0, 'clean_fail': 0}

    pbar = tqdm(total=args.max_articles, unit='article', desc='Extracting', unit_scale=True)

    for _, elem in context:
        if elem.tag != f'{{{NS}}}page':
            continue

        total_pages += 1
        title = elem.find(f'{{{NS}}}title')
        ns_elem = elem.find(f'{{{NS}}}ns')
        redirect = elem.find(f'{{{NS}}}redirect')
        text_elem = elem.find(f'{{{NS}}}revision/{{{NS}}}text')

        title_text = title.text if title is not None else ''

        if redirect is not None:
            skipped['redirect'] += 1
            elem.clear()
            continue

        if ns_elem is None or ns_elem.text != '0':
            skipped['non_main'] += 1
            elem.clear()
            continue

        raw_text = text_elem.text if text_elem is not None and text_elem.text else ''
        if len(raw_text) < 100:
            skipped['short'] += 1
            elem.clear()
            continue

        cleaned = clean_wikitext(raw_text)
        if len(cleaned) < 50:
            skipped['clean_fail'] += 1
            elem.clear()
            continue

        record = {
            'title': title_text,
            'text': cleaned,
            'url': f'https://en.wikipedia.org/wiki/{title_text.replace(" ", "_")}',
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + '\n')
        count += 1
        pbar.update(1)
        pbar.set_postfix(title=title_text[:60])

        if count >= args.max_articles:
            break

        elem.clear()

    pbar.close()
    out_file.close()
    f.close()

    print(f'\nDone: {count} articles -> {args.output}')
    print(f'  Pages scanned:  {total_pages}')
    for reason, n in skipped.items():
        if n:
            print(f'  Skipped ({reason}): {n}')


if __name__ == '__main__':
    main()
