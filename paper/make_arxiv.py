"""Assemble a clean arXiv submission package and verify it builds standalone.

arXiv rejects or mangles submissions that carry build artefacts, unused figures
or absolute paths, so this copies only what the document actually references and
then compiles the result in isolation.

    python paper/make_arxiv.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MAIN = "weight_of_the_adjoint"
OUT = os.path.join(ROOT, "arxiv")
SRC = os.path.join(OUT, "src")

# arXiv accepts only these characters in file names.
SAFE = re.compile(r"^[A-Za-z0-9_+\-.,=]+$")


def fail(msg):
    print(f"  FAIL  {msg}")
    return 1


def main():
    tex_path = os.path.join(HERE, f"{MAIN}.tex")
    tex = open(tex_path, encoding="utf-8").read()

    if os.path.isdir(SRC):
        shutil.rmtree(SRC)
    os.makedirs(os.path.join(SRC, "figs"))

    # --- copy only the figures the document actually includes -----------------
    used = sorted(set(re.findall(r"\\includegraphics\[[^\]]*\]\{figs/([^}]+)\}", tex)))
    missing = []
    for f in used:
        s = os.path.join(HERE, "figs", f)
        if not os.path.exists(s):
            missing.append(f)
            continue
        shutil.copy2(s, os.path.join(SRC, "figs", f))
    shutil.copy2(tex_path, os.path.join(SRC, f"{MAIN}.tex"))

    print(f"figures referenced : {len(used)}")
    print(f"figures copied     : {len(used) - len(missing)}")
    if missing:
        return fail(f"missing figures: {missing}")

    # --- preflight checks against arXiv's stated requirements -----------------
    print("\npreflight")
    rc = 0

    for root, _, files in os.walk(SRC):
        for f in files:
            if not SAFE.match(f):
                rc |= fail(f"illegal character in file name: {f}")
            if f.startswith("."):
                rc |= fail(f"hidden file will be dropped on announcement: {f}")
    print("  ok    file names are arXiv-safe (a-z A-Z 0-9 _ + - . , =)")

    banned = (".aux", ".log", ".out", ".toc", ".lof", ".lot", ".dvi", ".synctex.gz")
    stray = [f for _, _, fs in os.walk(SRC) for f in fs if f.endswith(banned)]
    if stray:
        rc |= fail(f"build artefacts present: {stray}")
    else:
        print("  ok    no build artefacts (.aux/.log/.out/...)")

    if os.path.exists(os.path.join(SRC, f"{MAIN}.pdf")):
        rc |= fail("the compiled PDF must not be uploaded alongside the source")
    else:
        print("  ok    compiled PDF not included")

    if re.search(r"\\date\{[^}]*\\today", tex):
        rc |= fail("\\today in \\date -- arXiv rebuilds PDFs and the date will drift")
    else:
        print("  ok    no \\today in \\date")

    if re.search(r"\\includegraphics[^{]*\{(?:[A-Za-z]:|/)", tex):
        rc |= fail("absolute path in \\includegraphics")
    else:
        print("  ok    no absolute paths in figure includes")

    exts = {os.path.splitext(f)[1].lower() for f in used}
    if exts - {".png", ".jpg", ".jpeg", ".pdf"}:
        rc |= fail(f"figure formats {exts} are not all pdflatex-compatible")
    else:
        print(f"  ok    figure formats {sorted(exts)} suit pdflatex")

    if "\\bibliography{" in tex or "\\addbibresource" in tex:
        print("  note  document uses BibTeX/biblatex -- a .bbl must be shipped")
    else:
        print("  ok    bibliography is inline (thebibliography); no .bbl needed")

    # --- compile in isolation, exactly as arXiv will --------------------------
    print("\ncompiling in isolation (pdflatex x3)")
    for i in range(3):
        p = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-halt-on-error", f"{MAIN}.tex"],
                           cwd=SRC, capture_output=True, text=True)
        if p.returncode != 0:
            tail = "\n".join(p.stdout.splitlines()[-25:])
            return fail(f"pdflatex pass {i + 1} failed:\n{tail}")

    log = open(os.path.join(SRC, f"{MAIN}.log"), encoding="utf-8",
               errors="replace").read()
    pages = re.search(r"Output written.*?\((\d+) pages", log)
    undef = len(re.findall(r"Reference `[^']+' on page \d+ undefined", log))
    print(f"  ok    {pages.group(1) if pages else '?'} pages, "
          f"{undef} undefined reference(s)")
    if undef:
        rc |= fail("undefined references remain")

    # --- keep the reference PDF outside the package, strip the rest -----------
    ref = os.path.join(OUT, "reference_build.pdf")
    shutil.move(os.path.join(SRC, f"{MAIN}.pdf"), ref)
    for f in os.listdir(SRC):
        if f.endswith(banned):
            os.remove(os.path.join(SRC, f))
    print(f"  ok    reference PDF saved to {os.path.relpath(ref, ROOT)} "
          f"-- diff it against arXiv's preview")

    zpath = os.path.join(OUT, f"arxiv_{MAIN}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(SRC):
            for f in sorted(files):
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, SRC))

    size = os.path.getsize(zpath) / (1024 * 1024)
    print(f"\nwrote {os.path.relpath(zpath, ROOT)}  ({size:.2f} MB)")
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            print(f"    {n}")
    if size > 50:
        rc |= fail("over arXiv's 50 MB limit -- request an exception or shrink figures")
    else:
        print(f"  ok    {size:.2f} MB is within arXiv's 50 MB limit")

    # --- metadata for the web form -------------------------------------------
    print()
    rc |= subprocess.run([sys.executable,
                          os.path.join(HERE, "make_arxiv_metadata.py")]).returncode

    print("\nPACKAGE READY" if rc == 0 else "\nPROBLEMS FOUND")
    return rc


if __name__ == "__main__":
    sys.exit(main())
