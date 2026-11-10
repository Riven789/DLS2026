def latex_escape(s):
    return (s.replace("$","\\$").replace("&","\\&").replace("%","\\%").replace("#","\\#").replace("_","\\_")
             .replace("^","\\^{}").replace("~","\\~{}"))

def gen_appendix():
    out=[]
    out.append(r"""\section*{Reading list by section}
The papers behind each part of the deck, in the order they are discussed, with one line on why each is worth reading. Entries with an arXiv identifier are included in the paper archive; entries without one are listed with a URL. Items marked \emph{secondary source} are blog posts, model cards or commentary.
""")
    for num,name,slides in SECTIONS:
        out.append(f"\\subsection*{{{num}. {latex_escape(name)} ({slides})}}")
        out.append(r"\begin{itemize}[leftmargin=1.2em,itemsep=3pt]\sloppy")
        for e in [p for p in P if p['sec']==num]:
            cite=f"arXiv:{e['arxiv']}" if e['arxiv'] else (f"\\url{{{e['url']}}}" if e['url'] else "")
            out.append(f"\\item \\textbf{{{latex_escape(e['authors'])} ({e['year']}).}} \\textit{{{latex_escape(e['title'])}}}. {cite}. {latex_escape(e['why'])}")
        out.append(r"\end{itemize}")
    return "\n".join(out)+"\n"

def gen_csv():
    import csv, io
    buf=io.StringIO(); w=csv.writer(buf)
    w.writerow(["section","section_name","key","authors","year","title","arxiv_id","url","why"])
    names={n:name for n,name,_ in SECTIONS}
    for e in P:
        url=e['url'] or (f"https://arxiv.org/abs/{e['arxiv']}" if e['arxiv'] else "")
        w.writerow([e['sec'],names[e['sec']],e['key'],e['authors'],e['year'],e['title'],e['arxiv'] or "",url,e['why']])
    return buf.getvalue()

def gen_bib():
    out=[]
    for e in P:
        if e['arxiv']:
            out.append(f"@article{{{e['key']},\n  author = {{{e['authors']}}},\n  title = {{{e['title']}}},\n  year = {{{e['year']}}},\n  journal = {{arXiv preprint arXiv:{e['arxiv']}}},\n  eprint = {{{e['arxiv']}}},\n  archivePrefix = {{arXiv}},\n  url = {{https://arxiv.org/abs/{e['arxiv']}}}\n}}")
        else:
            out.append(f"@misc{{{e['key']},\n  author = {{{e['authors']}}},\n  title = {{{e['title']}}},\n  year = {{{e['year']}}},\n  howpublished = {{\\url{{{e['url']}}}}}\n}}")
    return "\n\n".join(out)+"\n"

def gen_script():
    names={n:name for n,name,_ in SECTIONS}
    lines=[f"#!/bin/bash",
           f"# Downloads the reading list for '{TITLE}' from arXiv and zips it.",
           f"# Usage:  bash fetch_papers.sh          (needs curl and zip; ~{sum(1 for e in P if e['arxiv'] or (e['url'] and ('.pdf' in e['url'].lower() or 'spec-final-pdf' in e['url'])))} PDFs; takes a few minutes)",
           "# arXiv asks for no more than one request every few seconds from scripts; the sleep below respects that.",
           "set -u",
           f'ROOT="$(cd "$(dirname "$0")" && pwd)/{ZIPBASE}"',
           'mkdir -p "$ROOT"',
           'fail=0; ok=0',
           'get() {  # get <dir> <filename> <url>',
           '  mkdir -p "$ROOT/$1"',
           '  if [ -s "$ROOT/$1/$2" ]; then ok=$((ok+1)); return; fi',
           '  if curl -sS -L --retry 3 --retry-delay 5 -A "fetch_papers.sh (reading list downloader)" -o "$ROOT/$1/$2" "$3" && [ -s "$ROOT/$1/$2" ] && head -c 4 "$ROOT/$1/$2" | grep -q "%PDF"; then',
           '    ok=$((ok+1)); echo "ok   $1/$2"',
           '  else',
           '    rm -f "$ROOT/$1/$2"; fail=$((fail+1)); echo "FAIL $1/$2  ($3)"',
           '  fi',
           '  sleep 3',
           '}',
           ""]
    for e in P:
        secdir=f"{e['sec']}_{names[e['sec']].lower().replace(' ','_').replace(':','').replace(',','')}"
        fname=f"{e['key']}.pdf"
        if e['arxiv']:
            lines.append(f'get "{secdir}" "{fname}" "https://arxiv.org/pdf/{e["arxiv"]}"')
        elif e['url'] and e['url'].lower().endswith('.pdf') or (e['url'] and 'Paper.pdf' in e['url']) or (e['url'] and 'spec-final-pdf' in e['url']):
            lines.append(f'get "{secdir}" "{fname}" "{e["url"]}"')
        else:
            lines.append(f'echo "manual: {secdir}/{e["key"]}  ->  {e["url"]}"   # not a direct PDF; open in a browser')
    lines += ["",
              'cp "$(dirname "$0")/reading_list.csv" "$(dirname "$0")/reading_list.bib" "$(dirname "$0")/README.md" "$ROOT/" 2>/dev/null',
              'echo "downloaded: $ok   failed: $fail"',
              f'( cd "$(dirname "$ROOT")" && rm -f {ZIPBASE}.zip && zip -qr {ZIPBASE}.zip "$(basename "$ROOT")" ) && echo "wrote $(dirname "$ROOT")/{ZIPBASE}.zip"']
    return "\n".join(lines)+"\n"

def gen_readme():
    names={n:name for n,name,_ in SECTIONS}
    out=[f"# Reading list: {TITLE}","",
         "Companion papers to the deck and its lecture notes, grouped by section.",
         "`reading_list.csv` and `reading_list.bib` hold the same list as data; `fetch_papers.sh` downloads",
         "every arXiv entry into one folder per section and zips the result (`bash fetch_papers.sh`).",
         "Entries without an arXiv identifier are listed with a URL and must be fetched by hand.",""]
    for num,name,slides in SECTIONS:
        out.append(f"## {num}. {name} ({slides})"); out.append("")
        for e in [p for p in P if p['sec']==num]:
            link=f"https://arxiv.org/abs/{e['arxiv']}" if e['arxiv'] else e['url']
            out.append(f"- **{e['authors']} ({e['year']}).** *{e['title']}.* {link}  \n  {e['why']}")
        out.append("")
    return "\n".join(out)

if False:
    import sys, os
    base=os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(base,"parts","06-reading-list.tex"),"w").write(gen_appendix())
    outdir=os.path.join(base,"papers_kit"); os.makedirs(outdir,exist_ok=True)
    open(os.path.join(outdir,"reading_list.csv"),"w").write(gen_csv())
    open(os.path.join(outdir,"reading_list.bib"),"w").write(gen_bib())
    open(os.path.join(outdir,"fetch_papers.sh"),"w").write(gen_script())
    open(os.path.join(outdir,"README.md"),"w").write(gen_readme())
    print(len(P),"entries;",sum(1 for p in P if p['arxiv']),"on arXiv")
