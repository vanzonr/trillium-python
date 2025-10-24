name=trillium-python

all: ${name}.pdf

handout: ${name}_handout.pdf

${name}.pdf: ${name}.md
	./mds -vv $<

${name}_handout.pdf: ${name}.md 
	./mds -vvt $<

clean:
	\rm -rf ${name}.pdf ${name}_handout.pdf ${name}.tex ${name}_handout.tex
