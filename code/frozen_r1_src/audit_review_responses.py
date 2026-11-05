"""Check complete source comments and thanks-first responses, without changing them."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def source_sections(text):
    result = {}
    for reviewer in (1, 2, 3):
        start = text.index('Reviewer: %d' % reviewer)
        end = text.index('Reviewer: %d' % (reviewer+1)) if reviewer < 3 else text.index('\nAssociate Editor\n', start)
        block = text[start:end].split('Comments to the Author', 1)[1]
        result['reviewer%d' % reviewer] = [line.strip() for line in block.splitlines() if line.strip()]
    tail = text.split('\nAssociate Editor\n', 1)[1].split('Comments to the Author:', 1)[1]
    result['associate_editor'] = [tail.strip()]
    return result


def arguments(text, command=r'\reviewcomment'):
    result, cursor = [], 0
    while True:
        pos = text.find(command+'{', cursor)
        if pos < 0:
            break
        begin, end, depth = pos+len(command)+1, pos+len(command)+1, 1
        while end < len(text) and depth:
            if text[end] in '{}':
                slashes, prev = 0, end-1
                while prev >= 0 and text[prev] == '\\':
                    slashes, prev = slashes+1, prev-1
                if slashes % 2 == 0:
                    depth += 1 if text[end] == '{' else -1
            end += 1
        if depth:
            raise ValueError('Unclosed comment argument')
        result.append(dict(start=begin, end=end-1, text=text[begin:end-1]))
        cursor = end
    return result


def quote_tex(text):
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '^': r'\textasciicircum{}', '~': r'\textasciitilde{}',
               'π': r'\ensuremath{\pi}', 'δ': r'\ensuremath{\delta}',
               'Δ': r'\ensuremath{\Delta}', '≤': r'\ensuremath{\leq}',
               '…': r'\ldots{}', '“': '``', '”': "''", '–': '--', '—': '---',
               '\u00a0': ' '}
    return ''.join(escapes.get(char, char) for char in text)


def canonical(text):
    # Exact words/math identifiers are retained. TeX rendering, whitespace,
    # equivalent quote/dash glyphs, and the requested semicolon-to-period style
    # normalization are ignored. No token sorting or token deletion is allowed.
    for tex, plain in [(r'\textasciicircum{}', '^'), (r'\textasciitilde{}', '~'),
                       (r'\textbackslash{}', '\\'), (r'\ldots{}', '…'),
                       (r'\pi', 'π'), (r'\delta', 'δ'), (r'\Delta', 'Δ'),
                       (r'\leq', '≤'), (r'\le', '≤')]:
        text = text.replace(tex, plain)
    text = re.sub(r'\$|\\ensuremath', '', text)
    text = re.sub(r'\\([&%$#_{}])', r'\1', text)
    text = text.replace('{', '').replace('}', '')
    text = text.replace('---', '—').replace('--', '–')
    text = text.replace('``', '"').replace("''", '"').replace('“', '"').replace('”', '"')
    text = re.sub(r';\s*([a-z])', lambda match: '. ' + match.group(1).upper(), text)
    text = text.replace(';', '.')
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def audit(source, manuscript):
    comments = source_sections(source.read_text(encoding='utf-8-sig'))
    records = []
    for section, originals in comments.items():
        path = manuscript/'responses'/(section+'.tex')
        tex = path.read_text(encoding='utf-8')
        extracted = arguments(tex)
        responses = re.findall(r'\\begin\{response\}(.*?)\\end\{response\}', tex, re.S)
        if len(originals) != len(extracted) or len(originals) != len(responses):
            raise ValueError('Unmatched number of source comments/responses: '+section)
        for index, (original, actual, reply) in enumerate(zip(originals, extracted, responses)):
            records.append(dict(section=section, ordinal=index,
                complete_source_comment=canonical(original)==canonical(actual['text']),
                thanks_first=bool(re.match(r'\s*Thank you\b', reply)),
                comment_sha256=hashlib.sha256(original.encode()).hexdigest(),
                response_words=len(reply.split())))
    return dict(total=len(records), passed=len(records)==35 and all(
        r['complete_source_comment'] and r['thanks_first'] for r in records),
        punctuation_normalization='Reviewer semicolons may be rendered as periods. No words are removed.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), comments=records)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('../../comments.txt'))
    p.add_argument('--manuscript', type=Path, default=Path('../manuscript'))
    p.add_argument('--out', type=Path, default=Path('results/revision/review_response_audit.json'))
    args=p.parse_args()
    result=audit(args.source,args.manuscript)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
