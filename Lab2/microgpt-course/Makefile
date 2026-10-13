# Build all three documents into build/ and copy the PDFs here.
# Requires TeX Live with latexmk, biber, beamer (metropolis), tcolorbox, pgfplots, listings.
DOCS = lecture_notes slides study_guide lab2
all: $(DOCS:%=%.pdf)
%.pdf: %.tex refs.bib microgpt.py loss.csv
	latexmk -pdf -interaction=nonstopmode -outdir=build $<
	cp build/$@ $@
clean:
	rm -rf build
.PHONY: all clean
