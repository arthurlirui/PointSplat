% LaTeX source for PointSplat+ ICCV 2027 submission
%
% Requirements:
%   - ICCV 2027 LaTeX style file (iccv.sty) - place in this directory
%   - pdflatex for compilation
%
% Compile:
%   pdflatex main && bibtex main && pdflatex main && pdflatex main
%
% Paper structure:
%   main.tex          - Main paper (two-column, ICCV format)
%   references.bib    - Bibliography
%   iccv.sty          - ICCV style file (not included, download from ICCV website)
%
% The paper covers four extension directions:
%   1. Automatic Segmentation (Section 3.2)
%   2. Iterative Optimization (Section 3.3)
%   3. Edge Quantization (Section 3.4)
%   4. SLAM Integration (Section 3.5)

## Notes

- The iccv.sty file should be obtained from the ICCV 2027 author kit
- If iccv.sty is not available, the paper can be compiled with a generic
  article class by commenting out the \usepackage{iccv} line
- All experimental numbers marked with * are estimates pending full evaluation
