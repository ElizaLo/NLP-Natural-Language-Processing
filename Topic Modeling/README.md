# Topic Modeling

Topic modelling is an unsupervised machine learning method that **helps us discover hidden semantic structures in a paper**, that allows us to learn topic representations of papers in a corpus. The model can be applied to any kinds of labels on documents, such as tags on posts on the website.

> Dataset for this task:
  > - [A Million News Headlines](https://www.kaggle.com/therohk/million-headlines) - News headlines published over a period of 18 Years 
  
- [Topic Modeling with LSA, PLSA, LDA & lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)

##  Latent Dirichlet Allocation (LDA)

- [Topic Modelling in Python with NLTK and Gensim](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
- [Topic Modeling and Latent Dirichlet Allocation (LDA) in Python](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)
- [Topic Modelling in Python with NLTK and Gensim](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
- [Using LDA Topic Models as a Classification Model Input](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28)

### The Process

- We pick the number of topics ahead of time even if we’re not sure what the topics are.
- Each document is represented as a distribution over topics.
- Each topic is represented as a distribution over words.


### Code examples

- [LDA_news_headlines](https://github.com/susanli2016/NLP-with-Python/blob/master/LDA_news_headlines.ipynb)


## Guided Latent Dirichlet Allocation (LDA)

`GuidedLDA` OR `SeededLDA` implements latent Dirichlet allocation (LDA) using collapsed Gibbs sampling. `GuidedLDA` can be guided by setting some seed words per topic. Which will make the topics converge in that direction.

- [GuidedLDA’s documentation](https://guidedlda.readthedocs.io/en/latest/)
- [GuidedLDA](https://github.com/vi3k6i5/guidedlda) GitHub repository
- [How we Changed Unsupervised LDA to Semi-Supervised GuidedLDA](https://www.freecodecamp.org/news/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164/) Article

### Code examples

- [guided_LDA](https://github.com/ramirezfranco/guided_LDA_example/blob/master/guided_LDA.ipynb)
