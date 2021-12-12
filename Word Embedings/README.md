# Word Embedings


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

## Tokenization

### Readady Solutions

| Title | Description, Information |
| :---:         |          :--- |
|[Tokenizers](https://github.com/huggingface/tokenizers)|Fast State-of-the-Art Tokenizers optimized for Research and Production by [HuggingFace](https://huggingface.co/)|

## Bag of Words(BoW)

### Count Vectorizer

CountVectorizer tokenizes(tokenization means dividing the sentences in words) the text along with performing very basic preprocessing. It removes the punctuation marks and converts all the words to lowercase.
The vocabulary of known words is formed which is also used for encoding unseen text later.
An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document. The image below shows what I mean by the encoded vector.

> Count Vectorizer sparse matrix representation of words. (a) is how you visually think about it. (b) is how it is really represented in practice.

The row of the above matrix represents the document, and the columns contain all the unique words with their frequency. In case a word did not occur, then it is assigned zero correspondings to the document in a row.
Imagine it as a one-hot encoded vector and due to that, it is pretty obvious to get a sparse matrix with a lot of zeros.

- [10+ Examples for Using CountVectorizer](https://kavita-ganesan.com/how-to-use-countvectorizer/#.YGMSUy1c5WO)
