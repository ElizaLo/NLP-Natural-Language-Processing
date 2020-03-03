# Shallow Discourse Parsing

1. Make sure you have [Python 3](https://www.python.org/downloads/) and your favorite development environment installed on your laptop.
2. Install [NLTK](http://www.nltk.org/install.html). Follow the installation guide on the web page.
3. Install [the basic discourse parser](https://github.com/rknaebel/discopy/security/advisories). Follow the installation guide on the web page.
4. Install the visualization package [seaborn](https://seaborn.pydata.org/installing.html). Follow the installation guide on the web page.
Download or clone [the workshop repository](https://github.com/TScheffler/2020grammarly_ws). See the "Clone or download" button on the right.

- [Connective-Lex.info](http://connective-lex.info)


# Workshop on Shallow Discourse Parsing at Grammarly, Kyiv, 2020 by [Tatjana Scheffler](https://github.com/TScheffler)

[Workshop announcement](https://grammarly.ai/compling-workshop-shallow-discourse-parsing/)

## Schedule

10:15-13:30 - Part 1: The theory of shallow discourse structure

13:30-14:30 - lunch

14:30-17:45 - Part 2: Shallow Discourse parsing

## Preparation

Please download or clone this repository.

Please download/clone and install the baseline PDTB parser [discopy](https://github.com/rknaebel/discopy).

The scripts in this repository (and in the parser) use Python3, and need the NLTK package installed. We will be working in a Jupyter notebook.

Finally, we will be using the following dataset in two formats (passwords are distributed separately). Please note that the data must be deleted after completion of the workshop.
- [data, format 1](https://boxup.uni-potsdam.de/index.php/s/MxSceyJFraNz5cm)
- [data, format 2](https://boxup.uni-potsdam.de/index.php/s/7Y462x0PtMeTdBq)



## Part 1: The theory of shallow discourse structure

### Theory of discourse coherence

[slides](https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/Shallow%20Discourse%20Parsing/disc-parsing-workshop-slides-compressed.pdf)

### Discourse Annotations

*In-class exercise*: What is a connective? [Identifying connectives in Ukranian](uk_connectives/)

[PDTB exploration](pdtb_exploration/)

## Part 2: Shallow discourse parsing

### Introduction to shallow discourse parsing

[Error analysis on existing parsers](https://github.com/TScheffler/2019ESSLLI-discparsing/tree/master/day3#existing-discourse-parsers-for-english_)

### Implementing a discourse parser

Modify/play around with a connective detection module: [discopy](https://github.com/rknaebel/discopy)
