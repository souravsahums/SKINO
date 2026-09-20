# arXiv submission — exact steps

Paper: *The Weight of the Adjoint: Which Symplectic Form a Neural Operator
Preserves, and Whether It Survives the Lift*

Everything you need is in this `arxiv/` folder. Work top to bottom.

| File | What it is |
| --- | --- |
| `arxiv_weight_of_the_adjoint.zip` | **This is what you upload.** |
| `src/` | The unpacked contents of that zip, for inspection. |
| `metadata.txt` | Copy/paste answers for every web-form field. |
| `abstract.txt` | Abstract alone, flattened to plain text. |
| `reference_build.pdf` | What the source compiles to locally. Diff against arXiv's preview. |
| `SUBMISSION_STEPS.md` | This file. Do not upload it. |

Rebuild everything at any time with one command:

```powershell
$env:PATH = "C:\texlive\2026\bin\windows;$env:PATH"
python paper\make_arxiv.py
```

It refuses to produce a zip unless the package compiles standalone with zero
undefined references, so a green run means arXiv's AutoTeX will almost
certainly succeed too.

---

## Step 0 — Before you start (do this first; it can block you for days)

1. **Account.** Log in at <https://arxiv.org/user>. If you do not have an
   account, register with your `@microsoft.com` address — an institutional
   domain matters for step 0.3.
2. **Author identity.** Add your ORCID under *Account → ORCID*. It links this
   paper to your author page permanently and costs a minute now.
3. **Endorsement — already held, nothing to do.** Endorsement attaches to your
   account per category, not per paper: it is required only "before submitting
   their first paper to arXiv or a new category." You are endorsed for `cs.LG`
   from a previous submission, and a rejected paper does not revoke it —
   arXiv revokes endorsement only for policy violations. `math.NA` and
   `physics.comp-ph` are cross-lists, which do not require endorsement.
   Two caveats that *do* follow from a prior decline:
   - Moderation policy states that submitters with previously delayed or
     declined works "should anticipate closer scrutiny on future
     submissions." Budget the full one-to-four day moderation window.
   - If the declined paper was an earlier version of this work, moderators may
     treat this as a revision rather than a new submission and ask that the two
     be consolidated or versioned. Appendix A retracts our own earlier
     theorem, which makes the relationship visible. Pre-empt it in the
     **Comments** field: `Supersedes and corrects an earlier unannounced
     submission; see Appendix A.`
4. **Co-author consent.** You are submitting on behalf of Gareth O'Brien as
   well. Have his explicit sign-off on the final PDF before step 5 — after
   announcement the only way to change anything is a new version.
5. **Decide the license now** (step 4.7 makes it irrevocable). Recommendation:
   **CC BY 4.0**. It is the most permissive option that still requires
   attribution, and it is what TMLR and most ML venues expect. The
   arXiv-only perpetual license is the restrictive default — avoid it.

---

## Step 1 — Verify the package locally

```powershell
cd "c:\Users\sahusourav\OneDrive - Microsoft\Desktop\AdvancedNeuroOperator"
$env:PATH = "C:\texlive\2026\bin\windows;$env:PATH"
python paper\make_arxiv.py
```

You want to see all of this:

```
preflight
  ok    file names are arXiv-safe (a-z A-Z 0-9 _ + - . , =)
  ok    no build artefacts (.aux/.log/.out/...)
  ok    compiled PDF not included
  ok    no \today in \date
  ok    no absolute paths in figure includes
  ok    figure formats ['.png'] suit pdflatex
  ok    bibliography is inline (thebibliography); no .bbl needed
compiling in isolation (pdflatex x3)
  ok    22 pages, 0 undefined reference(s)
...
PACKAGE READY
```

Then **open `arxiv/reference_build.pdf` and actually read it.** Check the
title page, that all 12 figures render, and that nothing is stranded after the
references. This is your ground truth for step 4.

Why each preflight check exists, if you ever need to explain it:

- arXiv only accepts `a-z A-Z 0-9 _ + - . , =` in filenames, and they are
  case-sensitive.
- `.aux`, `.log`, `.out`, `.toc`, `.lof`, `.lot`, `.dvi` and the compiled
  `.pdf` must **not** be in the upload. AutoTeX regenerates them, and a stale
  `.aux` produces wrong cross-references in the announced PDF.
- `\today` in `\date` would make the date drift every time arXiv rebuilds the
  paper. Ours is `\date{}`, so this is already safe.
- Absolute paths in `\includegraphics` break on arXiv's build machines.
- Our bibliography is an inline `thebibliography` environment, so there is
  **no `.bib` and no `.bbl` to ship.** That removes the single most common
  cause of arXiv build failures. (If you ever switch to BibTeX/biblatex, you
  must upload the `.bbl` — arXiv does not run BibTeX for you.)

---

## Step 2 — Start the submission

1. Go to <https://arxiv.org/user>.
2. Click **START NEW SUBMISSION**.
3. Select **I am an author of this paper** (not a proxy submission).
4. Accept the submission agreement.

The system walks you through six numbered stages. Do not open a second
submission if you get confused — use the back navigation, or unsubmit later
(step 6).

---

## Step 3 — Upload and file check

1. On the **Prepare Files** stage, choose **Upload a compressed archive**.
2. Select `arxiv\arxiv_weight_of_the_adjoint.zip` (1.56 MB — far under the
   50 MB limit, so no size exception is needed).
3. Click **Check Files**.

What should happen:

- arXiv detects the archive contains LaTeX source and selects the **TeX/LaTeX**
  processor automatically.
- It identifies `weight_of_the_adjoint.tex` as the top-level file, because it
  is the only `.tex` in the root and the only one with `\documentclass`.
- The `figs/` subdirectory is preserved. AutoTeX always compiles from the root
  of the submission, which is why `\includegraphics{figs/...}` resolves.
- It should report **no files removed**. If it lists deletions, something
  slipped into the zip — go back to step 1 rather than proceeding.

> **Do not upload the PDF.** arXiv accepts either LaTeX source *or* a PDF, and
> for a TeX-authored paper it requires the source. Uploading the PDF instead
> gets the submission put on hold.

4. Click **Accept and Continue**, then **Confirm**.

---

## Step 4 — Read the compile log, then the preview

This is the step people skip and regret.

1. arXiv compiles the source and shows a **processing log**. Read it.
   - Warnings are normal. Errors are not.
   - Expect a note that it used a specific TeX Live year. arXiv's default is
     **TeX Live 2025**; we build locally against 2026. Our preamble uses only
     long-stable packages (`geometry`, `amsmath`, `amsthm`, `mathtools`,
     `booktabs`, `natbib`, `hyperref`, `caption`, `tikz`, `placeins`), so this
     should be a non-event — but it is exactly the kind of difference the
     preview will expose.
2. Click **View PDF** / **Preview**. **You cannot complete the submission
   without doing this, and you should not want to.**
3. Compare the preview against `arxiv/reference_build.pdf`, specifically:
   - 22 pages.
   - The three TikZ architecture diagrams (Figures 2, 3, 4) render — TikZ is
     the most version-sensitive thing in the document.
   - All nine PNG plots appear and are not stranded past the references.
   - No `??` or `[?]` anywhere — those are broken cross-references.
   - The notation table in §3.1 sits inline, not floated to the end.

If the preview is wrong, go back and fix the source. Do not proceed hoping it
resolves itself.

---

## Step 5 — Metadata

Open `arxiv/metadata.txt` and paste field by field.

| Field | Value |
| --- | --- |
| **Title** | The Weight of the Adjoint: Which Symplectic Form a Neural Operator Preserves, and Whether It Survives the Lift |
| **Authors** | `Sourav Sahu, Gareth O'Brien` |
| **Abstract** | Paste from `arxiv/abstract.txt` (1716 chars; the cap is 1920) |
| **Primary category** | `cs.LG` — Machine Learning |
| **Cross-list** | `math.NA` — Numerical Analysis; `physics.comp-ph` — Computational Physics |
| **Comments** | `22 pages, 12 figures, 7 tables. Code, full result set and figure generators: https://github.com/souravsahums/SKINO` |
| **Report number** | leave blank |
| **Journal ref / DOI** | leave blank — these are added *after* publication, without creating a new version |
| **MSC class** (optional) | `65M99, 68T07, 37K05` |
| **ACM class** (optional) | `I.2.6, G.1.8` |
| **License** | CC BY 4.0 |

Rules the form enforces that are easy to trip over:

- **Authors** must be `Forename Surname`, comma-separated. Do not add
  affiliations or emails in this field — they belong in the PDF, which already
  has them.
- **Title and abstract must be plain text.** A restricted TeX subset renders,
  but anything exotic shows up as literal source in listings and RSS. The
  abstract in `abstract.txt` has already been flattened for this
  (`W^{-1}K^{\top}W` → `W^-1K^TW`, and so on).
- **Do not put line-break formatting in the abstract.** Paste it as one
  paragraph.
- `cs.LG` as primary is right for the audience. `math.NA` is the honest
  cross-list given the weighted-adjoint / quadrature content, and
  `physics.comp-ph` reaches the PDE-surrogate readership. Cross-lists are
  reviewed by moderators; all three are defensible here.

---

## Step 6 — Submit, then check it immediately

1. Click **Submit**.
2. You will get a confirmation email with a **submission identifier**
   (e.g. `submit/1234567`). This is *not* the arXiv ID. The real arXiv ID is
   assigned only at announcement and cannot be issued in advance.
3. **Go straight back to <https://arxiv.org/user> and open your submission.**
   Re-read the title, authors and abstract as they now render.

### Announcement timing

Announcements run Sunday–Thursday evenings; nothing is announced Friday or
Saturday. Deadlines are US Eastern:

| Submitted between | Announced | Public |
| --- | --- | --- |
| Mon 14:00 – Tue 14:00 | Tue 20:00 | Tue night / Wed morning |
| Tue 14:00 – Wed 14:00 | Wed 20:00 | Wed night / Thu morning |
| Wed 14:00 – Thu 14:00 | Thu 20:00 | Thu night / Fri morning |
| Thu 14:00 – Fri 14:00 | Sun 20:00 | Sun night / Mon morning |
| Fri 14:00 – Mon 14:00 | Mon 20:00 | Mon night / Tue morning |

Check the live clock and next deadline at <https://arxiv.org/localtime>.

Note the Thursday-afternoon cliff: submitting at Thu 14:01 ET costs you three
days versus submitting at Thu 13:59 ET.

Separately, moderation quality checks can take **one to four days**, sometimes
longer, and can delay announcement independently of the schedule above. You
will be emailed if anything is flagged.

---

## Step 7 — If you find a mistake

**Before announcement:** on your user page, use the **Unsubmit** icon. That
returns the submission to *incomplete* status so you can edit and resubmit via
**Update**. **Delete** removes it entirely. Either is free and leaves no trace.

**Same day, before 14:00 ET:** edits do not create a new version stamp and do
not delay announcement.

**Same day, after 14:00 ET but before announcement:** edits still do not create
a version stamp, but they **irrevocably delay announcement** to the next cycle.

**After announcement:** you cannot edit. Use **Replace** to post v2. In the
`Comments:` field, state what changed *and re-paste the original comments* —
they are overwritten, not merged. For example:

```
22 pages, 12 figures, 7 tables; corrected typos in Section 5, added references.
Code: https://github.com/souravsahums/SKINO
```

Do **not** withdraw a paper because it was updated or published — use Replace
or add a journal reference respectively. Withdrawal notices are public and
permanent.

For a new version, regenerate the package with `python paper\make_arxiv.py`
and upload the fresh zip. Replace no more than once a week; after v5,
revisions stop appearing in the daily mailings.

---

## Step 8 — After it is live

1. Add the arXiv ID to the repo `README.md`.
2. If you submit to TMLR, put the arXiv ID in the submission — TMLR permits
   and expects public preprints.
3. Once published, add the **Journal ref** and **DOI** on arXiv. Those two
   fields are the only metadata you can update without creating a new version.

---

## Quick reference

| Item | Value |
| --- | --- |
| Upload | `arxiv/arxiv_weight_of_the_adjoint.zip` |
| Size | 1.56 MB (limit 50 MB) |
| Contents | 1 `.tex` + 9 PNGs in `figs/` |
| Top-level file | `weight_of_the_adjoint.tex` |
| Processor | pdflatex (auto-detected) |
| `.bbl` needed | No — inline `thebibliography` |
| Pages | 22 |
| Primary / cross-list | `cs.LG` / `math.NA`, `physics.comp-ph` |
| License | CC BY 4.0 |
| Rebuild command | `python paper\make_arxiv.py` |
