# Word Embeddings

## Classification of Word Embeddings

> **Vector space model** or **term vector model** is an algebraic model for representing text documents (and any objects, in general) as vectors of identifiers (such as index terms). It is used in **_information filtering, information retrieval, indexing and relevancy rankings_**. 
> **Semantics:** In linguistics, semantics is the subfield that studies meaning. Semantics can address meaning at the levels of words, phrases, sentences, or larger units of [discourse](https://en.wikipedia.org/wiki/Discourse). 

### Count based Vector space model (Non-semantic)

- Bag-of-words model
- 

### Other

- Ohe-hot encoding

## 

🔹 [Everything about Embeddings](https://medium.com/@b.terryjack/nlp-everything-about-word-embeddings-9ea21f51ccfe)

- [TF-IDF Vectorizer](https://github.com/ElizaLo/NLP-Natural-Language-Processing/tree/master/Word%20Embedings/TF-IDF%20Vectorizer)

  1. ### **_Word Embedings_**
      - **One-hot encoding**
      - **Feature vectors**
      - **Cooccurrence vectors**
      - **Sparse Distributed Representations (SDRs)**
      ----------------
      - **Static Word Embeddings:**
        - **Continuous Bag-of-Words (CBOW)**
          - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
        - **Skip-gram Model**
          - [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
          - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
        - **Word2Vec**
          - [Word2Vec Tutorial](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
          - [Implementing Deep Learning Methods and Feature Engineering for Text Data: The Continuous Bag of Words (CBOW)](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)
        - **FastText**
          - [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
          - [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf)
          - [FASTTEXT.ZIP:COMPRESSING TEXT CLASSIFICATION MODELS](https://arxiv.org/pdf/1612.03651.pdf)
        - **GloVe: Global Vectors for Word Representation**
          - [GloVe: Global Vectors for Word Representation Article](https://nlp.stanford.edu/pubs/glove.pdf)
          - [GloVe](https://nlp.stanford.edu/projects/glove/)
       -------------------------
      - **Deep Neural Networks for Word Representations:** 
        - **Sequence to Sequence Model (Seq2Seq)**
          - [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
          - [Order Matters: Sequence to sequence for sets](https://arxiv.org/pdf/1511.06391.pdf)
          - [A Comparison of Sequence-to-Sequence Models for Speech Recognition](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0233.PDF)
          - [Sequence to Sequence – Video to Text](https://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf)
          - [Understanding Encoder-Decoder Sequence to Sequence Model](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)
      ----------------------------
      - **Contextualized (Dynamic) Word Embeddings (LM):**
         - CoVe (Contextualized Word-Embeddings)
         - CVT (Cross-View Training)
         - ELMO (Embeddings from Language Models)
            - [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf)
         - ULMFiT (Universal Language Model Fine-tuning)
         - BERT (Bidirectional Encoder Representations from Transformers)
            - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
            - [FROM Pre-trained Word Embeddings TO Pre-trained Language Models — Focus on BERT](https://medium.com/@adriensieg/from-pre-trained-word-embeddings-to-pre-trained-language-models-focus-on-bert-343815627598)
         - GPT & GPT-2 (Generative Pre-Training)
         - Transformer XL (meaning extra long)
         - XLNet (Generalized Autoregressive Pre-training)
         - ENRIE (Enhanced Representation through kNowledge IntEgration)
         - (FlairEmbeddings (Contextual String Embeddings for Sequence Labelling))
---------------------------
  2. ### **_Sentence Embedings_**
      - [How to obtain Sentence Vectors?](https://medium.com/explorations-in-language-and-learning/how-to-obtain-sentence-vectors-2a6d88bd3c8b)
      - [Deep-learning-free Text and Sentence Embedding](https://www.offconvex.org/2018/06/17/textembeddings/)
-------------------------------------------------------      
  3. ### **_Text / Document Embedings_**
      - [Document Embedding Techniques](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d)

-----------------------------------------------------------------------

## One-hot encodding

> Also known as _**“1-of-N”**_ encoding (meaning the vector is composed of a single one and a number of zeros).

In one hot encoding, every word (even symbols) which are part of the given text data are written in the form of vectors, constituting only of **1** and **0**. So one hot vector is a vector whose elements are only 1 and 0. 

**Each word is written or encoded as one hot vector, with each one hot vector being unique.** This allows the word to be identified uniquely by its one hot vector and vice versa, that is no two words will have same one hot vector representation.

### Pros and Cons

**Pros:**

- One-hot encoding ensures that machine learning does not assume that higher numbers are more important. For example, the value '8' is bigger than the value '1', but that does not make '8' more important than '1'. The same is true for words: the value 'laughter' is not more important than 'laugh'.
- One-Hot-Encoding has the advantage that the result is binary rather than ordinal and that everything sits in an orthogonal vector space.

**Cons:**

- Curse of **dimensionality**, which refers to all sorts of problems that arise with data in high dimensions. Even with relatively small eight dimensions, our example text requires exponentially large memory space. Most of the matrix is taken up by zeros, so useful data becomes sparse. Imagine we have a vocabulary of 50,000. (There are roughly a million words in English language.) Each word is represented with 49,999 zeros and a single one, and we need 50,000 squared = 2.5 billion units of memory space. Not computationally efficient.
- **Hard to extract meanings.** Each word is embedded in isolation, and every word contains a single one and N zeros where N is the number of dimensions. The resulting set of vectors do not say much about one another. If our vocabulary had _“orange”, “banana”_ and _“watermelon”_, we can see the similarity between those words, such as the fact that they are types of fruit, or that they usually follow some form of the verb _“eat”_. We can easily form a mental map or cluster where these words exist close to each other. But with one-hot vectors, all words are equal distance apart.
- Another disadvantage of one-hot encoding is that it produces **multicollinearity** among the various variables, lowering the model's accuracy.

### 📰 Articles

- [One-hot](https://en.wikipedia.org/wiki/One-hot)
- [](https://medium.com/analytics-vidhya/one-hot-encoding-of-text-data-in-natural-language-processing-2242fefb2148)
- [How to One Hot Encode Sequence Data in Python](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)
- [One Hot Encoding to treat Categorical data parameters](https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/)

### Implementation

- [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Keras: to_categorical-function]([https://keras.io/api/utils/#to_categorical](https://keras.io/api/utils/python_utils/#to_categorical-function))

## Bag-of-words model (BoW)

In this model, a text (such as a sentence or a document) is represented as the [bag (multiset)](https://en.wikipedia.org/wiki/Multiset) of its words, disregarding grammar and even word order but keeping multiplicity.

> **Count vectorizer** creates a matrix with documents and token counts (bag of terms/tokens) therefore it is also known as **document term matrix**.

CountVectorizer tokenizes(tokenization means dividing the sentences in words) the text along with performing very basic preprocessing. It removes the punctuation marks and converts all the words to lowercase.
The vocabulary of known words is formed which is also used for encoding unseen text later.
An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document. The image below shows what I mean by the encoded vector.

The row of the above matrix represents the document, and the columns contain all the unique words with their frequency. In case a word did not occur, then it is assigned zero correspondings to the document in a row.
Imagine it as a one-hot encoded vector and due to that, it is pretty obvious to get a sparse matrix with a lot of zeros.

### 📰 Articles

- [Bag of words model](https://thatascience.com/learn-machine-learning/bag-of-words/)
- [10+ Examples for Using CountVectorizer](https://kavita-ganesan.com/how-to-use-countvectorizer/#.YGMSUy1c5WO)

### Implmentation 

- [Counting words in Python with sklearn's CountVectorizer](https://investigate.ai/text-analysis/counting-words-with-scikit-learns-countvectorizer/)

## N-gram Language Models

> An n-gram model is a type of probabilistic [language model](https://en.wikipedia.org/wiki/Language_model) for predicting the next item in such a sequence in the form of a (n − 1)–order [Markov model](https://en.wikipedia.org/wiki/Markov_chain).

An N-Gram is a connected string of N-items from a sample of text or speech. The N-Gram could be comprised of large blocks of words, or smaller sets of syllables. N-Grams are used as the basis for functioning N-Gram models, which are instrumental in natural language processing as a way of predicting upcoming text or speech.

An n-gram model models sequences, notably natural languages, using the statistical properties of n-grams.

This idea can be traced to an experiment by Claude Shannon's work in information theory. Shannon posed the question: given a sequence of letters (for example, the sequence "for ex"), what is the likelihood of the next letter? From training data, one can derive a probability distribution for the next letter given a history of size _n_: _a = 0.4, b = 0.00001, c = 0, ...._; where the probabilities of all possible "next-letters" sum to 1.0.

More concisely, an n-gram model predicts the next word based on the sequence of previous words.

When used for language modelling, independence assumptions are made so that each word depends only on the last n − 1 word. This Markov model is used as an approximation of the true underlying language. This assumption is important because it massively simplifies the problem of estimating the language model from data. In addition, because of the open nature of language, it is common to group words unknown to the language model together.

Note that in a simple n-gram language model, the probability of a word, conditioned on some number of previous words (one word in a bigram model, two words in a trigram model, etc.) can be described as following a categorical distribution (often imprecisely called a "multinomial distribution").

### 📚 Books

- [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) chapter, _“Speech and Language Processing”_ by Daniel Jurafsky & James H. Martin, 2021.
  - [Nlp - 2.1 - Introduction to N-grams](https://www.youtube.com/watch?v=dkUtavsPqNA)

### 📰 Articles

- [N-Grams](https://deepai.org/machine-learning-glossary-and-terms/n-gram)
- [n-gram](https://en.wikipedia.org/wiki/N-gram), wikipedia

### Implementation

- [N-gram Language Model with NLTK](https://www.kaggle.com/code/alvations/n-gram-language-model-with-nltk), Kaggle notebook

## Tokenization

### Readady Solutions

| Title | Description, Information |
| :---:         |          :--- |
|[Tokenizers](https://github.com/huggingface/tokenizers)|Fast State-of-the-Art Tokenizers optimized for Research and Production by [HuggingFace](https://huggingface.co/)|

## Models

### Sentence Embedings

| Title | Description, Information |
| :---:         |          :--- |
|[SentEval: evaluation toolkit for sentence embeddings](https://github.com/facebookresearch/SentEval)|SentEval is a library for evaluating the quality of sentence embeddings. We assess their generalization power by using them as features on a broad and diverse set of "transfer" tasks. SentEval currently includes 17 downstream tasks. We also include a suite of 10 probing tasks which evaluate what linguistic properties are encoded in sentence embeddings. Our goal is to ease the study and the development of general-purpose fixed-size sentence representations.|
|[HMTL (Hierarchical Multi-Task Learning model)](https://github.com/huggingface/hmtl)|HMTL is a Hierarchical Multi-Task Learning model which combines a set of four carefully selected semantic tasks (namely Named Entity Recoginition, Entity Mention Detection, Relation Extraction and Coreference Resolution). The model achieves state-of-the-art results on Named Entity Recognition, Entity Mention Detection and Relation Extraction. Using [SentEval](https://github.com/facebookresearch/SentEval), we show that as we move from the bottom to the top layers of the model, the model tend to learn more complex semantic representation.|
