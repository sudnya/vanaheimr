
IMAGES = 

all: archaeopteryx.pdf

archaeopteryx.pdf: archaeopteryx.tex archaeopteryx.bib $(IMAGES)
	pdflatex -halt-on-error archaeopteryx.tex
	bibtex archaeopteryx
	pdflatex -halt-on-error archaeopteryx.tex
	pdflatex -halt-on-error archaeopteryx.tex
	
clean:
	rm -f *.pdf
	rm -f *.aux
	rm -f *.ps
	rm -f *.dvi
	rm -f *.log
	rm -f *.bbl
	rm -f *.blg	
	rm -f *.nav	
	rm -f *.out	
	rm -f *.snm	
	rm -f *.toc
