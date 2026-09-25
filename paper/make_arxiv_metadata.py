"""Produce the plain-text metadata arXiv's web form asks you to paste."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "arxiv")
tex = open(os.path.join(HERE, "weight_of_the_adjoint.tex"), encoding="utf-8").read()

body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)

# arXiv renders a limited TeX subset in abstracts; inline maths survives, the
# rest should be flattened so the field reads cleanly in listings and RSS.
subs = [
    (r"\\citep\{[^}]*\}", ""), (r"\\citet\{[^}]*\}", ""),
    (r"\\emph\{([^}]*)\}", r"\1"), (r"\\textbf\{([^}]*)\}", r"\1"),
    (r"\\textsc\{([^}]*)\}", r"\1"), (r"\\texttt\{([^}]*)\}", r"\1"),
    (r"\\sacheb\{\}", "SA-Cheb"), (r"\\ckino\{\}", "CKINO"),
    (r"\\Wc\b", "W_cheb"), (r"\\Wu\b", "W_unif"),
    # named symbols must be resolved before the catch-all macro strip below
    (r"\\top\b", "T"), (r"\\omega\b", "omega"),
    (r"\\lVert\s*", "||"), (r"\s*\\rVert", "||"),
    (r"\s*\\times\s*", "x"), (r"\\approx", "~"), (r"\\to\b", "->"),
    (r"\\,", " "), (r"\\;", " "), (r"---", " -- "), (r"--", "-"),
    (r"\\%", "%"), (r"\\&", "&"),
    (r"\$([^$]*)\$", r"\1"),
    (r"\\[a-zA-Z]+", ""),
    (r"[{}]", ""),
]
for pat, rep in subs:
    body = re.sub(pat, rep, body)
abstract = re.sub(r"\s+", " ", body).strip()

meta = f"""arXiv submission metadata -- paste these into the web form
=========================================================

TITLE
The Weight of the Adjoint: Which Symplectic Form a Neural Operator Preserves,
and Whether It Survives the Lift

AUTHORS  (arXiv wants "Forename Surname" separated by commas)
Sourav Sahu, Gareth O'Brien

ABSTRACT  ({len(abstract)} characters -- arXiv's limit is 1920)
{abstract}

PRIMARY CATEGORY
cs.LG  (Machine Learning)

CROSS-LIST
math.NA  (Numerical Analysis)
physics.comp-ph  (Computational Physics)

COMMENTS
25 pages, 12 figures, 9 tables. Code, full result set and figure generators:
https://github.com/souravsahums/SKINO

LICENSE  (recommended)
CC BY 4.0 -- arXiv.org perpetual non-exclusive license is the minimum, but
CC BY maximises reuse and is what most ML venues now expect.

MSC CLASS   (optional)   65M99, 68T07, 37K05
ACM CLASS   (optional)   I.2.6, G.1.8
"""

with open(os.path.join(OUT, "metadata.txt"), "w", encoding="utf-8") as f:
    f.write(meta)
with open(os.path.join(OUT, "abstract.txt"), "w", encoding="utf-8") as f:
    f.write(abstract + "\n")

print(f"abstract: {len(abstract)} chars "
      f"({'OK' if len(abstract) <= 1920 else 'TOO LONG -- arXiv caps at 1920'})")
print(f"wrote arxiv/metadata.txt and arxiv/abstract.txt\n")
print(abstract[:400] + " ...")
