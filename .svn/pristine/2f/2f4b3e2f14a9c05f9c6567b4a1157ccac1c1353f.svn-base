
IMAGES = 
NAME = vanaheimr

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex $(NAME).bib $(IMAGES)
	pdflatex -halt-on-error $(NAME).tex
	bibtex $(NAME)
	pdflatex -halt-on-error $(NAME).tex
	pdflatex -halt-on-error $(NAME).tex
	
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
