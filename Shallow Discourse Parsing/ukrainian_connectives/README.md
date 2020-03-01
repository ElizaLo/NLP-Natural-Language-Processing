# Group task: Creating an initial lexicon of Ukrainian connectives

1. Determine whether the given candidate word (or phrase) is a connective:
   (Note: There are also some Russian tokens in the list of candidates, based on wrongly aligned material in the underlying corpora. Please just exclude any non-Ukrainian words.)
   - it is a fixed (closed class) expression that cannot be modified or conjugated
   - it semantically relates to arguments
   - the arguments are "abstract objects" (propositions, facts, events, ...)
   - the arguments are in principle expressible as clauses

2. After determining the connectives (from 1.), group them into semantic classes:
   - temporal
   - contingency (causal + conditional)
   - contrast (contrast and similarity)
   - expansion (additive relations)

3. Group connectives by substitution tests: which connectives can replace each other in the examples?

4. Determine the syntactic and semantic class of each connective and enter it in the Google sheet:
   https://docs.google.com/spreadsheets/d/134_SxwVkoPpMlsyxZ2uThlaYBQ6rK0XxnmVpRKOjeNs/edit?usp=sharing

   *syntactic categories:*
   - coord_conj = coordinating conjunction
   - subord_conj = subordinating conjunction
   - adverb
   - preposition = pre- or postposition
   - other = other, for example phrasal connectives

   *semantic senses:*

   We use the simplified list of PDTB3 senses in [pdtb3-senses-simplified.txt](pdtb3-senses-simplified.txt).

   For more information, see the [PDTB3 Annotation Manual](https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf)

## Data

I've extracted the connective candidates from the bilingual (English-Ukrainian) dictionaries of the QED and OpenSubtitles corpora (see references below). The corpus data is available for **research purposes** only! Please see [OPUS](http://opus.nlpl.eu/index.php) for more information. 

As seeds, I have used the English connectives (heads only) extracted from the PDTB, as distributed in the [Eng-DiMLex lexicon](https://github.com/discourse-lab/en_dimlex/blob/master/en_dimlex.xml). 

Similarly, the corpus samples for each connective candidate have been extracted from the (EN-UK section of the) QED corpus. 

## References

- A. Abdelali, F. Guzman, H. Sajjad and S. Vogel, "The AMARA Corpus: Building parallel language resources for the educational domain", The Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC'14). Reykjavik, Iceland, 2014. Pp. 1856-1862. Isbn. 978-2-9517408-8-4.
- Debopam Das, Tatjana Scheffler, Peter Bourgonje and Manfred Stede. Constructing a Lexicon of English Discourse Connectives. In: K. Komatani, D. Litman, K. Yu, A. Papangelis, L. Cavedon, M. Nakano (eds.): Proceedings of the 19th Annual SIGdial Meeting on Discourse and Dialogue. Melbourne, Australia , July 2018
- P. Lison and J. Tiedemann, 2016, OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles. In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016) [http://www.opensubtitles.org/](http://www.opensubtitles.org/)
- J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)
