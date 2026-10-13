#!/usr/bin/env python3
import sys
from pathlib import Path
m=Path(sys.argv[1])
required=['tutorial.tex','tutorial.pdf','bibliography.bib','slides.tex','slides.pdf','discussion_questions.tex','discussion_questions.pdf','project_ideas.tex','project_ideas.pdf','understanding_check.tex','understanding_check.pdf','sources.md','QA.md']
missing=[x for x in required if not (m/x).is_file() or (m/x).stat().st_size==0]
if missing: raise SystemExit('missing: '+', '.join(missing))
for pdf in [x for x in required if x.endswith('.pdf')]:
  if (m/pdf).stat().st_size < 5000: raise SystemExit(f'undersized PDF: {pdf}')
print(f'PASS {m}: {len(required)} required artifacts present')
