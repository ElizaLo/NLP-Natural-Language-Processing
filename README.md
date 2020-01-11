# Natural Language Processing 

🔹 [100 Must-Read NLP Papers](http://masatohagiwara.net/100-nlp-papers/)

## 🔺 Projects:
  - [**_Spam Detection_**](https://github.com/ElizaLo/ML-with-Jupiter#spam-detection)
  - [**_Text Generator_**](https://github.com/ElizaLo/ML-with-Jupiter#text-generator)
  - **_Quora Insincere Questions Classification_**
  - [**_Question Answering System using BiDAF Model on SQuAD_**](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD)

## 🔺 Embedings

🔹 [Everything about Embeddings](https://medium.com/@b.terryjack/nlp-everything-about-word-embeddings-9ea21f51ccfe)

  1. **_Word Embedings_**
      - **One-hot encoding**
      - **Feature vectors**
      - **Cooccurrence vectors**
      - **Sparse Distributed Representations (SDRs)**
      - **Continuous Bag-of-Words (CBOW)**
        - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
      - **Continuous Skip-gram Model**
        - []()
        - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
      - **Word2Vec**
        - [Word2Vec Tutorial](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
        - [Implementing Deep Learning Methods and Feature Engineering for Text Data: The Continuous Bag of Words (CBOW)](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)
      - **GloVe: Global Vectors for Word Representation**
        - [GloVe: Global Vectors for Word Representation Article](https://nlp.stanford.edu/pubs/glove.pdf)
        - [GloVe](https://nlp.stanford.edu/projects/glove/)
      - **Sequence to Sequence Model (Seq2Seq)**
  2. **_Sentence Embedings_**
      - [How to obtain Sentence Vectors?](https://medium.com/explorations-in-language-and-learning/how-to-obtain-sentence-vectors-2a6d88bd3c8b)
      - [Deep-learning-free Text and Sentence Embedding](https://www.offconvex.org/2018/06/17/textembeddings/)
  3. **_Text / Document Embedings_**
      - [Document Embedding Techniques](https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d)
      
## 🔺 Models and Algorithms:      
  1. **_Levenshtein Distance_**
      - [Wikipedia](https://en.wikipedia.org/wiki/Levenshtein_distance)
      - [Levenshtein Distance Calculator](https://phiresky.github.io/levenshtein-demo/)
  2. **_Conditional Random Fields_**
      - [Overview of Conditional Random Fields](https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541)
  3. **_Formal Concept Analysis_**
      - [Formal Concept Analysis Examples](https://www.upriss.org.uk/fca/examples.html)
      - [FCA Online](https://fca-tools-bundle.com)
  4. **_Non-negative Matrix Factorization_**
      - [Wikipedia](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
      - **_Algorithms for Non-negative Matrix Factorization_**
        - [Algorithms for Non-negative Matrix Factorization](https://www.researchgate.net/publication/2538030_Algorithms_for_Non-negative_Matrix_Factorization) (Daniel D. **Lee** and H. Sebastian **Seung**) 
          - [Article at GitHub](https://github.com/ElizaLo/NLP/blob/master/Articles/Algorithms_for_Non-negative_Matrix_Factorization.pdf)
   5. **_Kullback–Leibler divergence (relative entropy)_**
       - [Wikipedia](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)
   6. **_Latent Semantic Analysis (LSA)_**
       - [Wikipedia](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
       - **_Probabilistic Latent Semantic Analysis (PLSA)_**
          - [Wikipedia](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis)
          - [Вероятностный латентный семантический анализ](http://www.machinelearning.ru/wiki/index.php?title=Вероятностный_латентный_семантический_анализ)
   7. **_Latent Dirichlet Allocation (LDA)_**
      - [Article](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) (David M. **Blei**, Andrew Y. **Ng**, Michael I. **Jordan**)
      - [Тематическое моделирование](http://www.machinelearning.ru/wiki/index.php?%20title=Тематическое_моделирование#.D0.9B.D0.B0.D1.82.D0.B5.%20D0.BD.D1.82.D0.BD.D0.BE.D0.B5_.D1.80.D0.B0.D0.B7.D0.BC.D0.%20B5.D1.89.D0.B5.D0.BD.D0.B8.D0.B5_.D0.94.D0.B8.D1.80.D0.B8.D1%20.85.D0.BB.D0.B5)
      - [Intuitive Guide to Latent Dirichlet Allocation](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158)
   8. **_Gibbs sampling_**
      - [Wikipedia](https://en.wikipedia.org/wiki/Gibbs_sampling)
      - [Topic modeling using Latent Dirichlet Allocation(LDA) and Gibbs Sampling explained!](https://medium.com/analytics-vidhya/topic-modeling-using-lda-and-gibbs-sampling-explained-49d49b3d1045)
   9. 

## 🔺 Courses:
  - [ ] [Natural Language Processing | Dan Jurafsky, Christopher](https://www.youtube.com/playlist?list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm)
  - [ ] [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
  - [ ] [Natural Language Processing | University of Michigan](https://www.youtube.com/playlist?list=PLLssT5z_DsK8BdawOVCCaTCO99Ya58ryR)


# Question Answering System using BiDAF Model on SQuAD

Implemented a Bidirectional Attention Flow neural network as a baseline on SQuAD, improving Chris Chute's model [implementation](https://github.com/chrischute/squad/blob/master/layers.py), adding word-character inputs as described in the original paper and improving [GauthierDmns' code](https://github.com/GauthierDmn/question_answering).

  - [Project Repository](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD)
  - [Paper](https://github.com/ElizaLo/NLP/blob/master/Question%20Answering%20System/Question%20Answering%20System%20based%20on%20SQuAD.pdf)
  - [Used Articles](https://github.com/ElizaLo/NLP/tree/master/Question%20Answering%20System/Articles)
  - [Useful Articles](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD#useful-articles)
  - [Useful Links](https://github.com/ElizaLo/Question-Answering-based-on-SQuAD#useful-links)
  
 # [Diploma](https://github.com/ElizaLo/NLP/tree/master/Diploma/Articles)
 
 ## Used [Articles](https://github.com/ElizaLo/NLP/tree/master/Diploma/Articles):

1. "**_A Tensor-based Factorization Model of Semantic Compositionality_**" (Tim Van de Cruys, Thierry Poibeau, Anna Korhonen)
2. "**_A Neural Network Approach to Selectional Preference Acquisition_**" (Tim Van de Cruys)
3. "**_A Generalisation of Lexical Functions for Composition in Distributional Semantics_**" (Antoine Bride, Tim Van de Cruys, Nicholas Asher)
4. "**_Mining Discourse Markers
for Unsupervised Sentence Representation Learning_**" (Damien Sileo, Tim Van De Cruys, Camille Pradel and Philippe Muller)
5. "**_Метод автоматического построения онтологических баз знаний. I. Разработка семантико-синтаксической модели есественного языка._**" (А. А. Марченко)
  
  # Articles
  
  - [ ] [Interpretation of Natural Language Rules in Conversational Machine Reading](https://arxiv.org/abs/1809.01494)
  - [ ] [Skip-Thought Vectors](https://github.com/ElizaLo/NLP/blob/master/University%20Course%20of%20NLP/Articles/5950-skip-thought-vectors.pdf), [Article](https://arxiv.org/abs/1506.06726)
  
 # Some Concepts
 
 - [x] **Selectional Preference** - (Katz and Fodor, 1963; Wilks, 1975; Resnik, 1993) are the tendency for a word to semantically select or constrain which other words may appear in a direct syntactic relation with it." In case this selection is expressed in binary term (**_allowed/not-allowed_**), it is also called selectional restriction (Séaghdha and Korhonen, 2014). SP can be contrasted with **_verb subcategorization_** "with subcategorization describing the syntactic arguments taken by a verb, and selectional preferences describing the semantic preferences verbs have for their arguments" (Van de Cruys et al., 2012)
    - [Selectional preference, Natural Language Understanding Wiki](https://natural-language-understanding.fandom.com/wiki/Selectional_preference)
 - [x] **Selectional Restrictions** - 
    - [Selectional Restrictions, Jurafsky](https://web.stanford.edu/~jurafsky/slp3/slides/22_select.pdf)
