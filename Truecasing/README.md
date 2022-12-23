# Truecasing

> **Proper noun** -  is a noun that identifies a single entity and is used to refer to that entity (_Africa, Jupiter, Sarah, Microsoft_) as distinguished from a **common noun**, which is a noun that refers to a class of entities (continent, planet, person, corporation) and may be used when referring to instances of a specific class (a continent, another planet, these persons, our corporation). Some proper nouns occur in plural form (optionally or exclusively), and then they refer to groups of entities considered as unique (_the Hendersons, the Everglades, the Azores, the Pleiades_). Proper nouns can also occur in secondary applications, for example modifying nouns (the Mozart experience; his Azores adventure), or in the role of common nouns (he's no Pavarotti; a few would-be _Napoleons_). The detailed definition of the term is problematic and, to an extent, governed by convention.
>
> - [Proper noun](https://en.wikipedia.org/wiki/Proper_noun), Wikipedia

There are several practical approaches to the Truecasing problem:

- **sentence segmentation:** splitting the input text into sentences and up-casing the first word of each sentence.
- **part-of-speech (POS) tagging:** looking up the definition and context of each word within the sentence, and up-casing words with specific tags, e.g. nouns.
- **name-entity-recognition (NER):** classifying words within a sentence into specific categories, and deciding to up-case e.g. person names, etc.
- **statistical modeling:** training a statistical model on words and a group of words that usually appear in capitalized format.

## Articles

- [How To Capitalize Words Using AI](https://towardsdatascience.com/how-to-capitalize-words-using-ai-13750444d459)
- [Truecasing in natural language processing](https://towardsdatascience.com/truecasing-in-natural-language-processing-12c4df086c21)
