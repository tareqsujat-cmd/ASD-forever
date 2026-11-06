"""Collect the figures the results document actually cites into a submission-ready tree.

Placement is derived from the document rather than declared here: a figure cited before the
"# Supplementary results" heading becomes a main figure, one cited after it becomes a
supplementary figure, and anything in the study's figure directory that the document never
cites is left behind and listed so the omission is visible rather than silent.

    python organise_paper_figures.py --summary RESULTS_SUMMARY.md --out paper_figures
"""
from __future__ import annotations

import argparse, re, shutil
from pathlib import Path

SUPP_MARK = "# Supplementary results"


def slug(caption: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", caption.lower()).strip("_")
    return re.sub(r"_+", "_", s)[:52]


def main() -> None:
    ap = argparse.ArgumentParser(description="Organise cited figures for submission")
    ap.add_argument("--summary", default="RESULTS_SUMMARY.md")
    ap.add_argument("--out", default="paper_figures")
    ap.add_argument("--study", default=None,
                    help="study directory, to report figures the document never cites")
    args = ap.parse_args()

    text = Path(args.summary).read_text()
    cut = text.find(SUPP_MARK)
    cut = len(text) if cut < 0 else cut

    out = Path(args.out)
    for sub in ("main", "supplementary"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*.png"):
            old.unlink()

    seen, rows = set(), []
    n_main = n_supp = 0
    for m in re.finditer(r"!\[([^\]]*)\]\(([^)]+\.png)\)", text):
        caption, src = m.group(1), Path(m.group(2))
        if not src.exists() or src.resolve() in seen:
            continue
        seen.add(src.resolve())
        is_main = m.start() < cut
        if is_main:
            n_main += 1
            name = f"Figure{n_main}_{slug(caption)}.png"
        else:
            n_supp += 1
            name = f"FigureS{n_supp}_{slug(caption)}.png"
        dest = out / ("main" if is_main else "supplementary") / name
        shutil.copy2(src, dest)
        rows.append((dest.relative_to(out), caption, src))

    lines = ["# Figures", "",
             f"Collected from `{args.summary}`. A figure's number follows the order in which "
             "the results document cites it, and its section — main or supplementary — is "
             "taken from where the citation falls, so this tree stays consistent with the "
             "text by construction. Regenerate with `python organise_paper_figures.py`.", "",
             "## Main figures", "",
             "| figure | caption | source |", "|---|---|---|"]
    for rel, cap, src in rows:
        if rel.parts[0] == "main":
            lines.append(f"| `{rel.name}` | {cap} | `{src}` |")
    lines += ["", "## Supplementary figures", "",
              "| figure | caption | source |", "|---|---|---|"]
    for rel, cap, src in rows:
        if rel.parts[0] == "supplementary":
            lines.append(f"| `{rel.name}` | {cap} | `{src}` |")

    if args.study:
        figdir = Path(args.study) / "figures"
        if figdir.is_dir():
            # the document cites numbered copies (figNN_<name>.png) that the summary
            # generator makes from these sources, so a source is "cited" when some cited
            # file carries its name as a suffix
            cited_names = {q.name for q in seen}
            def is_cited(name: str) -> bool:
                return any(c == name or c.endswith("_" + name) for c in cited_names)
            uncited = sorted(p.name for p in figdir.glob("*.png")
                             if p.resolve() not in seen and not is_cited(p.name))
            if uncited:
                lines += ["", "## Generated but not cited", "",
                          "These exist in the study's figure directory and are referenced "
                          "nowhere in the results document. They are listed rather than "
                          "copied, so that a figure dropped from the text is visible here "
                          "instead of disappearing.", ""]
                lines += [f"- `{u}`" for u in uncited]

    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(f"{n_main} main + {n_supp} supplementary -> {out}/")
    print(f"manifest: {out/'README.md'}")


if __name__ == "__main__":
    main()
