TECTONIC ?= $(HOME)/bin/tectonic

.PHONY: all verify clean
all: deck.pdf

deck.pdf: deck.tex
	$(TECTONIC) --keep-logs --keep-intermediates deck.tex

verify: deck.pdf
	@test "$$(pdfinfo deck.pdf | awk '/^Pages:/ {print $$2}')" = 20
	@! grep -Eqi '(^| )(! LaTeX Error|Undefined control sequence|Emergency stop|Fatal error)' deck.log
	@! grep -Ei 'Overfull \\vbox' deck.log
	@pdftotext deck.pdf - | grep -q 'Einstein Summation'
	@echo 'Verified: exact 20 pages, extractable text, no fatal errors or overfull vboxes.'

clean:
	rm -f deck.aux deck.log deck.nav deck.out deck.snm deck.toc deck.vrb deck.synctex.gz deck.pdf
