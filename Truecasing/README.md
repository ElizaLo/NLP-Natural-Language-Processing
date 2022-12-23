# Truecasing

**Truecasing**, also called **capitalization recovery**, **capitalization correction**, or **case restoration**, is the problem in natural language processing (NLP) of determining the proper [capitalization](https://en.wikipedia.org/wiki/Capitalization) of words where such information is unavailable. This commonly comes up due to the standard practice (in English and many other languages) of automatically capitalizing the first word of a sentence. It can also arise in badly cased or noncased text (for example, all-lowercase or all-uppercase text messages).

- [Letter case](https://en.wikipedia.org/wiki/Letter_case#Sentence_case)
- [Title case](https://en.wikipedia.org/wiki/Title_case)

> **Proper noun** -  is a noun that identifies a single entity and is used to refer to that entity (_Africa, Jupiter, Sarah, Microsoft_) as distinguished from a **common noun**, which is a noun that refers to a class of entities (continent, planet, person, corporation) and may be used when referring to instances of a specific class (a continent, another planet, these persons, our corporation). Some proper nouns occur in plural form (optionally or exclusively), and then they refer to groups of entities considered as unique (_the Hendersons, the Everglades, the Azores, the Pleiades_). Proper nouns can also occur in secondary applications, for example modifying nouns (the Mozart experience; his Azores adventure), or in the role of common nouns (he's no Pavarotti; a few would-be _Napoleons_). The detailed definition of the term is problematic and, to an extent, governed by convention.
>
> - [Proper noun](https://en.wikipedia.org/wiki/Proper_noun), Wikipedia

💠 There are several practical approaches to the Truecasing problem:

- **sentence segmentation:** splitting the input text into sentences and up-casing the first word of each sentence.
- **part-of-speech (POS) tagging:** looking up the definition and context of each word within the sentence, and up-casing words with specific tags, e.g. nouns.
- **name-entity-recognition (NER):** classifying words within a sentence into specific categories, and deciding to up-case e.g. person names, etc.
- **statistical modeling:** training a statistical model on words and a group of words that usually appear in capitalized format.
- A **spell checker** can be used to identify words that are always capitalized.

## 🛠️ Frameworks

| Title | Description, Information |
| :---:         |          :--- |
|[TrueCase](https://pypi.org/project/truecase/)|<p>A language independent, statistical, language modeling based tool in Python that restores case information for text.</p><p>The model was inspired by the paper of [Lucian Vlad Lita et al., tRuEcasIng](https://www.cs.cmu.edu/~llita/papers/lita.truecasing-acl2003.pdf) but with some simplifications.</p><p>A model trained on NLTK English corpus comes with the package by default, and for other languages, a script is provided to create the model. This model is not perfect, train the system on a large and recent dataset to achieve the best results (e.g. on a recent dump of Wikipedia).</p><ul><li> :octocat: [TrueCase](https://github.com/daltonfury42/truecase)</li><li> 📄 [tRuEcasIng](https://www.cs.cmu.edu/~llita/papers/lita.truecasing-acl2003.pdf)</li></ul>|
|[TrueCaseAnnotator](https://stanfordnlp.github.io/CoreNLP/truecase.html), [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) by StanfordNLP|<p>Recognizes the “true” case of tokens (how it would be capitalized in well-edited text) where this information was lost, e.g., all upper case text. This is implemented with a discriminative model using the CRF sequence tagger.</p><ul><li> :octocat: [Stanford CoreNLP](https://github.com/stanfordnlp/CoreNLP)</li><li> 🤗 [stanfordnlp/CoreNLP ](https://huggingface.co/stanfordnlp/CoreNLP/tree/main)</li></ul>|
|[Language Independet Truecaser for Python](https://github.com/nreimers/truecaser)|<p>This is an implementation of a trainable Truecaser for Python.</p><p>A truecaser converts a sentence where the casing was lost to the most probable casing. Use cases are sentences that are in all-upper case, in all-lower case or in title case.</p><p>A model for English is provided, achieving an accuracy of **98.39%** on a small test set of random sentences from Wikipedia.</p><p>The model was inspired by the paper of [Lucian Vlad Lita et al., tRuEcasIng](https://www.cs.cmu.edu/~llita/papers/lita.truecasing-acl2003.pdf) but with some simplifications.</p>|

## Papers

- [Papers with Code - Truecasing](https://paperswithcode.com/search?q_meta=&q_type=&q=Truecasing)

## 📰 Articles

- [How To Capitalize Words Using AI](https://towardsdatascience.com/how-to-capitalize-words-using-ai-13750444d459)
- [Truecasing in natural language processing](https://towardsdatascience.com/truecasing-in-natural-language-processing-12c4df086c21)
