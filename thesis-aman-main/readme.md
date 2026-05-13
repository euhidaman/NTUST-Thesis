# NTUST Thesis Template
> This template is modified from [NTUST Thesis Template 1.7.1](https://www.overleaf.com/latex/templates/ntust-thesis-template-1-dot-7-1-english-version/xngjwvkqkrgj) in overleaf

## Files that need to be modified
> You should change the content in files starting with numbers.
> New chapter files go in `100_sections/` and must be added to `main.tex` with `\subfile{}`.
- 000_frontpages
  > Don't forget to replace `01_recommendation_form.pdf` and `02_qualification_form.pdf` with yours!
- 100_sections
- 200_appendices
- BIB/reference.bib
- FIG/<figures>

## Files you do not need to modify
- `matters/` — front and back page layout templates
- `watermark/` — watermark image files
- `ntust_report.cls` — document class
- `common_env.tex` — shared layout settings

## How to compile
Requires **XeLaTeX** (not pdfLaTeX).
```
xelatex main.tex
```

## How to turn off the watermark
In `main.tex`, comment out `\watermarktrue` and uncomment `\watermarkfalse`:
```latex
% \watermarktrue
\watermarkfalse
```
