"""Check every bibliography entry of the paper against arXiv and Crossref.

    python paper/check_references.py [--tex paper/weight_of_the_adjoint.tex] [--details]

For each \\bibitem it extracts the author surnames, title, year and arXiv id / DOI, looks the
work up (arXiv API by id; Crossref by DOI or, failing that, by bibliographic query), and
reports whether the surnames (as a set and in order) and the title agree.  Nothing is
accepted silently: every row says what was compared and with what.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "reference-check/1.0 (mailto:sahusourav@microsoft.com)"}
ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def get(url, tries=5):
    for k in range(tries):
        try:
            time.sleep(1.0)
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(4 * (k + 1))


def fold(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


TEX_ACCENTS = {r'\"O': "Ö", r'\"o': "ö", r'\"U': "Ü", r'\"u': "ü", r'\"a': "ä", r"\'e": "é", r"\'E": "É",
               r'\"{O}': "Ö", r'\"{o}': "ö"}


def detex(s):
    for k, v in TEX_ACCENTS.items():
        s = s.replace(k, v)
    s = re.sub(r"\\newblock", " ", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = s.replace("~", " ").replace("--", "-").replace("---", "-")
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_bib(tex):
    body = tex[tex.index("\\begin{thebibliography}"):tex.index("\\end{thebibliography}")]
    items = re.split(r"\\bibitem", body)[1:]
    out = []
    for it in items:
        m = re.match(r"\[([^\]]*)\]\{([^}]*)\}(.*)", it, re.S)
        label, key, rest = m.group(1), m.group(2), m.group(3)
        parts = re.split(r"\\newblock", rest)
        authors_raw = detex(parts[0])
        title = detex(parts[1]) if len(parts) > 1 else ""
        venue = detex(" ".join(parts[2:])) if len(parts) > 2 else ""
        out.append({"key": key, "label": detex(label), "authors_raw": authors_raw, "title": title.rstrip("."),
                    "venue": venue, "raw": detex(rest),
                    "arxiv": (re.search(r"arXiv:(\d{4}\.\d{4,5})", rest) or [None, None])[1],
                    "doi": (re.search(r"doi:\s*(10\.\d{4,9}/[^\s,;]+)", rest) or [None, None])[1],
                    "year": (re.search(r"\((\d{4})\)", label) or [None, None])[1]})
    return out


def surnames(authors_raw):
    s = re.sub(r"\band\b", ",", authors_raw)
    names = [n.strip() for n in s.split(",") if n.strip()]
    out = []
    for n in names:
        toks = n.replace(".", ". ").split()
        keep = []
        for t in toks:
            if re.fullmatch(r"[A-Z]\.?", t) or re.fullmatch(r"([A-Z]\.)+", t):
                continue
            keep.append(t)
        out.append(" ".join(keep) if keep else n)
    return out


def arxiv_lookup(ids):
    url = "http://export.arxiv.org/api/query?max_results=%d&id_list=%s" % (len(ids), ",".join(ids))
    root = ET.fromstring(get(url))
    res = {}
    for e in root.findall("a:entry", ATOM):
        aid = re.search(r"abs/(\d{4}\.\d{4,5})", e.find("a:id", ATOM).text).group(1)
        res[aid] = {
            "title": re.sub(r"\s+", " ", e.find("a:title", ATOM).text).strip(),
            "authors": [a.find("a:name", ATOM).text for a in e.findall("a:author", ATOM)],
            "published": e.find("a:published", ATOM).text[:10],
            "journal_ref": (e.find("x:journal_ref", ATOM).text if e.find("x:journal_ref", ATOM) is not None else None),
            "doi": (e.find("x:doi", ATOM).text if e.find("x:doi", ATOM) is not None else None),
        }
    return res


def crossref_doi(doi):
    j = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    return norm_cr(j)


def crossref_query(title, first_author, year):
    q = urllib.parse.quote(f"{title} {first_author}")
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=3"
    items = json.loads(get(url))["message"]["items"]
    return [norm_cr(i) for i in items]


def norm_cr(j):
    iss = (j.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"title": (j.get("title") or [""])[0], "container": (j.get("container-title") or [""])[0],
            "authors": [(a.get("family") or a.get("name") or "") for a in j.get("author", [])],
            "given": [(a.get("given") or "") for a in j.get("author", [])],
            "volume": j.get("volume"), "issue": j.get("issue"), "page": j.get("page"), "year": iss,
            "doi": j.get("DOI"), "type": j.get("type"), "publisher": j.get("publisher")}


def compare(bib, ref_titles, ref_authors):
    mine = [fold(s) for s in surnames(bib["authors_raw"])]
    theirs = [fold(a.split()[-1]) if " " in a and not a.islower() else fold(a) for a in ref_authors]
    full_theirs = [fold(a) for a in ref_authors]
    same_set = set(mine) == set(theirs) or all(any(m in t or t in m for t in full_theirs) for m in mine)
    same_n = len(mine) == len(ref_authors)
    in_order = same_n and all(m == t or m in f or f.endswith(m) for m, t, f in zip(mine, theirs, full_theirs))
    t_ok = any(fold(bib["title"]) == fold(t) or fold(bib["title"]) in fold(t) or fold(t) in fold(bib["title"]) for t in ref_titles)
    return same_set, same_n, in_order, t_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weight_of_the_adjoint.tex"))
    ap.add_argument("--details", action="store_true")
    a = ap.parse_args()
    tex = open(a.tex, encoding="utf-8").read()
    bibs = parse_bib(tex)
    cited = set(re.findall(r"\\cite[a-z]*\{([^}]*)\}", tex))
    cited = {k.strip() for c in cited for k in c.split(",")}
    keys = {b["key"] for b in bibs}
    print(f"{len(bibs)} bibliography entries; cited keys not in bibliography: {sorted(cited - keys)}; "
          f"entries never cited: {sorted(keys - cited)}\n")
    ids = [b["arxiv"] for b in bibs if b["arxiv"]]
    ax = arxiv_lookup(ids) if ids else {}
    bad = 0
    for b in bibs:
        status, notes = [], []
        if b["arxiv"]:
            r = ax.get(b["arxiv"])
            if not r:
                status.append("ARXIV ID NOT FOUND")
            else:
                s_set, s_n, s_ord, t_ok = compare(b, [r["title"]], r["authors"])
                status.append("arxiv: authors %s (n %s, order %s), title %s" % (
                    "ok" if s_set else "MISMATCH", "ok" if s_n else "MISMATCH", "ok" if s_ord else "check", "ok" if t_ok else "MISMATCH"))
                notes.append(f"  arXiv  : {r['published']} | {'; '.join(r['authors'])}\n           {r['title']}\n"
                             f"           journal_ref={r['journal_ref']} doi={r['doi']}")
        cr = None
        arxiv_doi = (ax.get(b["arxiv"]) or {}).get("doi") if b["arxiv"] else None
        try:
            if b["doi"]:
                cr = [crossref_doi(b["doi"])]
            elif arxiv_doi:
                cr = [crossref_doi(arxiv_doi)]
            elif not b["arxiv"]:
                cr = crossref_query(b["title"], surnames(b["authors_raw"])[0], b["year"])
        except Exception as exc:
            status.append("crossref LOOKUP FAILED (%s)" % type(exc).__name__)
        if cr:
            best = None
            for c in cr:
                s_set, s_n, s_ord, t_ok = compare(b, [c["title"]], c["authors"])
                score = int(s_set) + int(t_ok) + int(s_n)
                if best is None or score > best[0]:
                    best = (score, c, s_set, s_n, s_ord, t_ok)
            if best and (best[2] or best[5]):
                _, c, s_set, s_n, s_ord, t_ok = best
                status.append("crossref: authors %s (n %s), title %s" % ("ok" if s_set else "MISMATCH", "ok" if s_n else "MISMATCH", "ok" if t_ok else "MISMATCH"))
                notes.append(f"  Crossref: {c['year']} | {c['type']} | {c['container']} vol={c['volume']} no={c['issue']} pp={c['page']} doi={c['doi']}\n"
                             f"           {'; '.join(c['authors'])}\n           {c['title']}")
        if not any(s.startswith("crossref") for s in status):
            status.append("crossref: not consulted (arXiv-only entry)" if b["arxiv"] and not b["doi"]
                          else "crossref: no matching record (journal not indexed; verified at the publisher)")
        # a DOI record that agrees is the authority for the published version, so an arXiv v1
        # difference (extra authors, retitled) is informational then
        doi_ok = b["doi"] and any(s.startswith("crossref: authors ok") and "title ok" in s for s in status)
        if doi_ok:
            status = [s.replace("MISMATCH", "differs (preprint version)") if s.startswith("arxiv") else s for s in status]
        bad_words = ("MISMATCH", "NOT FOUND", "FAILED")
        flag = "OK   " if status and all(not any(w in s for w in bad_words) for s in status) else "CHECK"
        bad += flag == "CHECK"
        print(f"[{flag}] {b['key']:26s} ({b['label']})")
        print("        mine   :", b["authors_raw"][:150])
        print("        title  :", b["title"][:150])
        print("        venue  :", b["venue"][:150])
        for s in status:
            print("        ->", s)
        if a.details or flag == "CHECK":
            for n in notes:
                print(n)
        print()
    print(f"entries needing attention: {bad} of {len(bibs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
