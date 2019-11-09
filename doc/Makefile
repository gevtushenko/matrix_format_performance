MAINFILE = post

all: $(MAINFILE).pdf

$(MAINFILE).pdf: $(MAINFILE).tex
	pdflatex -shell-escape $(MAINFILE).tex
	pdflatex -shell-escape $(MAINFILE).tex

.PHONY: clean
clean:
	-rm -f *.aux
	-rm -f *.log
	-rm -f *.toc
	-rm -f *.bbl
	-rm -f *.blg
	-rm -f *.out
	-rm -f make/bib

.PHONY: cleanall
cleanall: clean
	-rm -f *.pdf
	-rm -f *.ps
	-rm -rf ./make
