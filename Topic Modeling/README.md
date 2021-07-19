# Topic Modeling


Topic modelling is an unsupervised machine learning method that **helps us discover hidden semantic structures in a paper**, that allows us to learn topic representations of papers in a corpus. The model can be applied to any kinds of labels on documents, such as tags on posts on the website.

> Ещё можно называть мягкой би-кластеризацией

- [Topic Modeling](https://paperswithcode.com/task/topic-models) on Papers with Code
- [Topic Modeling](https://stackoverflow.com/questions/tagged/topic-modeling) on StackOverflow


## Models

- **Unsupervised**
	- Latent Dirichlet Allocation (LDA)
	- Expectation–maximization algorithm (EM Algorithm)
	- Probabilistic latent semantic analysis (PLSA)
	- LDA2Vec
	- Sentence-BERT (SBERT)
- **Sepervised or semi-supervised**
	- Guided Latent Dirichlet Allocation (Guided LDA)
	- Anchored CorEx: Hierarchical Topic Modeling with Minimal Domain Knowledge (CorEx)


> Dataset for this task:
  > - [A Million News Headlines](https://www.kaggle.com/therohk/million-headlines) - News headlines published over a period of 18 Years 
  
## Papers

- [Latent Dirichlet Allocation (LDA) and Topic modeling: models, applications, a survey](https://paperswithcode.com/paper/171104305/review/)
- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
- [Incorporating Lexical Priors into Topic Models](https://aclanthology.org/E12-1021.pdf)

## Articles 

- [My First Foray into Data Science: Semi-Supervised Topic Modeling](https://www.nuwavesolutions.com/my-first-foray-into-data-science-semi-supervised-topic-modeling/)
	- **Some key points:**
		-  Taking 100 records from each topic to create an evenly distributed training data set increased my accuracy by at least 10%. 
		-  Here are a few other hyperparameters to tune (alpha, beta, etc.), and I can always gather a larger quasi-stratified sample to throw at the model (keeping in mind that I can’t be sure exactly how evenly distributed it is).
		-  Some other ideas that have cropped up are leveraging a **concept ontology** (or **word embedding**) to enhance the depth of my seed words, synthetically duplicating the curated documents to increase the size of the test set (to make a training set for supervised learning), or applying **transfer learning** from a large, external corpus and hope that the topics align with the internal business topics. And, of course, there’s the world of **deep learning**.
		-  Other things I plan to try with my data set (other than continue to lobby for more, better data) are **hierarchical agglomerative clustering**, multiple individual binary classifiers, and a series of hierarchical classifiers (we have learned that certain topics are linked to certain countries, which we have been able to tag with >90% accuracy in these same documents). The most promising seems to be using business knowledge to narrow down the number of possible topics (from 26 to 10 or 11) and then attempt the classification.
		-  So each document was tagged (albeit potentially incorrectly) with multiple tags. It ranged from 2 to 10 tags per document. When I was attempting my stratified sampling (100 records per topic), I selected documents that had only 2 tags. My assumption here was that if a document only had two topics, it was going to be more specific for the topic. Using that logic, I selected 100 records per topic, where each document had two topics. I also ensured that I was doing sampling without replacement, so there was no possibility that the model was learning the same subset of frequent terms for different topics. **_For example,_** I can talk at a high level about science and politics and sports, but if I’m only talking about science, then I’m more likely to use topic-specific words more frequently. This would help bump the relative frequency of topic-specific terms, helping my model learn more clearly.
- [Topic Modeling with Gensim (Python)](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)
- [Topic Modeling with LSA, PLSA, LDA & Lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)
- [Topic Modeling with Gensim (Python)](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#18dominanttopicineachsentence)
- [Topic Modeling in Python](https://ourcodingclub.github.io/tutorials/topic-modelling-python/)
- [Topic modeling visualization – How to present the results of LDA models?](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
- [2 latent methods for dimension reduction and topic modeling](https://towardsdatascience.com/2-latent-methods-for-dimension-reduction-and-topic-modeling-20ff6d7d547)
	- Latent Semantic Analysis (LSA)
	- Latent Dirichlet Allocation (LDA)
- [Topic Modelling using Word Embeddings and Latent Dirichlet Allocation](https://medium.com/analytics-vidhya/topic-modelling-using-word-embeddings-and-latent-dirichlet-allocation-3494778307bc)
	- Clustering using ‘wordtovec’ embeddings
	- Clustering using LDA ( Latent Dirichlet Analysis)
- [Latent Dirichlet Allocation for Beginners: A high level intuition](https://medium.com/@pratikbarhate/latent-dirichlet-allocation-for-beginners-a-high-level-intuition-23f8a5cbad71)
- [Topic modeling made just simple enough.](https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/)
- [How Stuff Works: A Comprehensive Topic Modelling Guide with NMF, LSA, PLSA, LDA & lda2vec (Part-2)](https://medium.com/@souravboss.bose/comprehensive-topic-modelling-with-nmf-lsa-plsa-lda-lda2vec-part-2-e3921e712f11)
- [Topic Modeling with Latent Dirichlet Allocation](https://towardsdatascience.com/topic-modeling-with-latent-dirichlet-allocation-e7ff75290f8)

______________________________________________________________

- классификация и категоризация документов
- тематическая сегментация документов
- атоматическое аннотирование документов
- автоматическая суммаризация коллекций

## Предварительная обработка и очистка текстов

- удаление форматирования и переносов
- удаление обрывочной и нетекстовой информации
- исправление опечаток
- сливание коротких текстов
- Формирование словаря:
	- Лемматизация
	- Стемминг
	- Выделение терминов (term extraction)
	- Удаление stopwords и слишком редких слов (реже 10 раз)
	- Использовать биграммы

### Базовые предположения

- Порядок слов в документе не важен (bag of words)
- Каждая пара **_(d, w)_** связана с некторой темой **_t ∈ T_** 
- Гипотеза условной независимости (cлова в документах генерируются темой, а не документом): **_p(w | t, d) = p(w | t)_**
- Иногда вводят предположение о разрежености:
	- Документ относится к небольшому числу тем
	- Тема состоит из небольшого числа терминов 
- Документ **_d_** -- это смесь распределений **_p(w | t)_** с весами **_p(t | d)_**

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_1.png" width="681" height="376">

### Постанвка задачи

**Дано:**
- **_W_** - словарь терминов (слов или словосочетаний)
- **_D_** - коллекция текстовых документов **_d ⊂ W_**
- **_n <sub>dw</sub>_** - сколько раз термин (слово) **_w_** встретится в документе **_d_**
- **_n <sub>d</sub>_** - длина документа **_d_**

**Найти:**
Параметры вероятностной тематической модели (формула полной вероятности):

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_2.png" width="280" height="72">

Условные распределения:
- **_φ<sub>w t</sub> = p(w | t)_** - вероятности терминов **_w_** в каждой теме **_t_**
- **_Θ<sub>t d</sub> = p(t | d)_** - вероятности тем **_t_** в каждом документе **_d_**

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_3.png" width="581" height="488">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_4.png" width="501" height="231">

**Стохастическая матрица - столбцы дискретные вероятностные распределения**
- В **_Φ_** неотрицательные нормарованные значения, сумма по каждому столбцу = 1, столбец - дискретное распределение вероятностей

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_5.png" width="600" height="473">

### Не одно решение

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_6.png" width="594" height="171">

**Решение: Использование регуляризаторов**

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_7.png" width="594" height="469">

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_8.png" width="590" height="500">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_9.png" width="482" height="156">

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_16.png" width="597" height="460">


##  Latent Dirichlet Allocation (LDA)

- **Generative model**

### Articles

- [Topic Modelling in Python with NLTK and Gensim](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
- [Topic Modeling and Latent Dirichlet Allocation (LDA) in Python](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)
- [Topic Modelling in Python with NLTK and Gensim](https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21)
- [Using LDA Topic Models as a Classification Model Input](https://towardsdatascience.com/unsupervised-nlp-topic-models-as-a-supervised-learning-input-cf8ee9e5cf28)

### The Process

- We pick the number of topics ahead of time even if we’re not sure what the topics are.
- Each document is represented as a distribution over topics.
- Each topic is represented as a distribution over words.

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_10.png" width="598" height="337">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_11.png" width="601" height="428">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_12.png" width="577" height="291">

### Two important parameters of the algorithm are:

- **Topic concentration / Beta**
- **Document concentration / Alpha**

For the symmetric distribution, a high alpha-value means that each document is likely to contain a mixture of most of the topics, and not any single topic specifically. A low alpha value puts less such constraints on documents and means that it is more likely that a document may contain mixture of just a few, or even only one, of the topics. Likewise, a high beta-value means that each topic is likely to contain a mixture of most of the words, and not any word specifically, while a low value means that a topic may contain a mixture of just a few of the words.

If, on the other hand, the distribution is asymmetric, a high alpha-value means that a specific topic distribution (depending on the base measure) is more likely for each document. Similarly, high beta-values means each topic is more likely to contain a specific word mix defined by the base measure.

In practice, a high alpha-value will lead to documents being more similar in terms of what topics they contain. A high beta-value will similarly lead to topics being more similar in terms of what words they contain.



### Code examples

- [LDA_news_headlines](https://github.com/susanli2016/NLP-with-Python/blob/master/LDA_news_headlines.ipynb)


<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_15.png" width="606" height="502">


# Semi-Supervised Topic Modeling

## Guided Latent Dirichlet Allocation (LDA)

`GuidedLDA` OR `SeededLDA` implements latent Dirichlet allocation (LDA) using collapsed Gibbs sampling. `GuidedLDA` can be guided by setting some seed words per topic. Which will make the topics converge in that direction.

- [Incorporating Lexical Priors into Topic Models](https://www.aclweb.org/anthology/E12-1021.pdf) Paper
- [GuidedLDA’s documentation](https://guidedlda.readthedocs.io/en/latest/)
- [GuidedLDA](https://github.com/vi3k6i5/guidedlda) GitHub repository
- [How we Changed Unsupervised LDA to Semi-Supervised GuidedLDA](https://www.freecodecamp.org/news/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164/) Article

### Articles 

- [How did I tackle a real-world problem with GuidedLDA?](https://medium.com/analytics-vidhya/how-i-tackled-a-real-world-problem-with-guidedlda-55ee803a6f0d)
  - [Guided LDA_6topics-4-grams](https://github.com/ShahrzadH/Insight_Project_SHV/blob/master/notebook/Guided%20LDA_6topics-4-grams.ipynb) Code Exapmle
- [Labeled LDA + Guided LDA topic modelling](https://stackoverflow.com/questions/54814727/labeled-lda-guided-lda-topic-modelling)

### Code examples

- [Topic modelling using guided lda](https://www.kaggle.com/nvpsani/topic-modelling-using-guided-lda) Kaggle
- [guided_LDA](https://github.com/ramirezfranco/guided_LDA_example/blob/master/guided_LDA.ipynb)


### Issues

- [**Installation Error**](https://github.com/vi3k6i5/GuidedLDA/issues/26)

**Solution:**

Running on OS X 10.14 w/Python3.7. able to get past them by upgrading cython:

`pip3 install -U cython`

Removing the following 2 lines from `setup.cfg`:

`[sdist]`
```@python
pre-hook.sdist_pre_hook = guidedlda._setup_hooks.sdist_pre_hook
```

And then running the original installation instructions:

```@shell
git clone https://github.com/vi3k6i5/GuidedLDA
cd GuidedLDA
sh build_dist.sh
python setup.py sdist
pip3 install -e .
```
- **The package `GuidedLDA` doesn't installed**

**Solution:**

The package that this is built of off is LDA and it installed with no issue. I managed to copy from the GuidedLDA package: the guidedlda.py, utils.py, datasets.py and the few NYT data set items into the original LDA package, post installation.

[GuidedLDA_WorkAround](https://github.com/dex314/GuidedLDA_WorkAround)

1. Pull down the repository.

2. Install the original LDA package. 
	https://pypi.org/project/lda/
	
3. Drop the *.py files from the GuidedLDA_WorkAround repo in the lda folder under site-packages for your specific enviroment.

4. Profit...


## Anchored CorEx: Hierarchical Topic Modeling with Minimal Domain Knowledge (2017)

**Cor**relation **Ex**planation (**CorEx**) is a topic model that yields rich topics that are maximally informative about a set of documents. The advantage of using CorEx versus other topic models is that it can be easily run as an unsupervised, semi-supervised, or hierarchical topic model depending on a user's needs. For semi-supervision, CorEx allows a user to integrate their domain knowledge via **"anchor words"**. This integration is flexible and allows the user to guide the topic model in the direction of those words. This allows for creative strategies that promote topic representation, separability, and aspects. More generally, this implementation of CorEx is good for clustering any sparse binary data.

- **Paper**: Gallagher, R. J., Reing, K., Kale, D., and Ver Steeg, G. [**"Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge"**](https://www.transacl.org/ojs/index.php/tacl/article/view/1244). Transactions of the Association for Computational Linguistics (TACL), 2017.
- [GitHub repository](https://github.com/gregversteeg/corex_topic)
- [Example Notebook](https://github.com/gregversteeg/corex_topic/blob/master/corextopic/example/corex_topic_example.ipynb)
- [PyPI page for CorEx](https://pypi.org/project/corextopic/)
- [Anchored Correlation Explanation: Topic Modeling With Minimal Domain Knowledge (TACL) : Ryan J. Gallagher, Kyle Reing, David Kal](https://vimeo.com/276403824) at ACL Performance


### Modifications of CorEx

- [Bio CorEx: recover latent factors with Correlation Explanation (CorEx)](https://github.com/gregversteeg/bio_corex)
- [Correlation Explanation Methods](https://github.com/hrayrhar/T-CorEx)
	- Linear CorEx
	- T-CorEx
- [Latent Factor Models Based on Linear Total Correlation Explanation (CorEx)](https://github.com/gregversteeg/LinearCorex)


### Useful Articles 

- [Interactive Search using BioBERT and CorEx Topic Modeling](https://www.kaggle.com/jdparsons/biobert-corex-topic-search)

### Issues on GitHub

- Update original model with the new data → https://github.com/gregversteeg/corex_topic/issues/31
- Test the model on new data → https://github.com/gregversteeg/corex_topic/issues/24
- How change the model, in particular, recalculation of probability estimates document-topic → https://github.com/gregversteeg/corex_topic/issues/33
- 

## Regulariazation

### Kullback–Leibler divergence (relative entropy)

Способ померять растояние между двумя распределениями

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_13.png" width="618" height="456">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_14.png" width="613" height="189">

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_17.png" width="614" height="500">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_18.png" width="602" height="552">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_19.png" width="571" height="188">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_20.png" width="605" height="507">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_21.png" width="595" height="493">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_22.png" width="610" height="329">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_23.png" width="610" height="515">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_24.png" width="593" height="430">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_25.png" width="598" height="484">

## Other models, tools and possible solutions

- [tomotopy](https://github.com/bab2min/tomotopy)
	- > tomotopy is a Python extension of tomoto (Topic Modeling Tool) which is a Gibbs-sampling based topic model library written in C++. It utilizes a vectorization of modern CPUs for maximizing speed. The current version of tomoto supports several major topic models
	- [tomotopy API documentation (v0.11.1)](https://bab2min.github.io/tomotopy/v0.11.1/en/) 
- [BigARTM](https://bigartm.readthedocs.io/en/stable/)
	- > BigARTM is a powerful tool for topic modeling based on a novel technique called Additive Regularization of Topic Models. This technique effectively builds multi-objective models by adding the weighted sums of regularizers to the optimization criterion. BigARTM is known to combine well very different objectives, including sparsing, smoothing, topics decorrelation and many others. Such combination of regularizers significantly improves several quality measures at once almost without any loss of the perplexity.
	- [BigARTM Documentation](https://bigartm.readthedocs.io/_/downloads/en/v0.8.3/pdf/)
	- [GitHub](https://github.com/bigartm/bigartm)
	- [Пример использования библиотеки BigARTM для построения тематической модели](https://www.coursera.org/lecture/unsupervised-learning/primier-ispol-zovaniia-bibliotieki-bigartm-dlia-postroieniia-tiematichieskoi-IOfXa)
	- [Hierarchical topic modeling with BigARTM library](https://towardsdatascience.com/hierarchical-topic-modeling-with-bigartm-library-6f2ff730689f)
- Stochastic block model
	- [A network approach to topic models](https://advances.sciencemag.org/content/4/7/eaaq1360)
	- [Bayesian Core-Periphery Stochastic Block Models](https://ryanjgallagher.github.io/code/sbms/overview)
		- > Core-periphery structure is one of the most ubiquitous mesoscale patterns in networks. This code implements two Bayesian stochastic block models in Python for modeling hub-and-spoke and layered core-periphery structures. It can be used for probabilistic inference of core-periphery structure and model selection between the two core-periphery characterizations.
		- [GitHub](https://github.com/ryanjgallagher/core_periphery_sbm)
	- [TopSBM: Topic Models based on Stochastic Block Models](https://topsbm.github.io)
	- [hSBM_Topicmodel](https://github.com/martingerlach/hSBM_Topicmodel)
		- >  A tutorial for topic-modeling with hierarchical stochastic blockmodels using graph-tool.
	- [A scikit-learn extension for Topic Models based on Stochastic Block Models](https://github.com/TopSBM/topsbm)
	- 
- [Neural Topic Model](https://github.com/zll17/Neural_Topic_Models)
- [LDA2vec: Word Embeddings in Topic Models](https://www.datacamp.com/community/tutorials/lda2vec-topic-model)
	- [Introducing our Hybrid lda2vec Algorithm](https://multithreaded.stitchfix.com/blog/2016/05/27/lda2vec/#topic=38&lambda=1&term=)
	- [lda2vec: Tools for interpreting natural language](https://github.com/cemoody/lda2vec)
	- [lda2vec-tf](https://github.com/meereeum/lda2vec-tf)
		- > `TensorFlow` implementation of Christopher Moody's lda2vec, a hybrid of Latent Dirichlet Allocation & word2vec
	- [lda2vec](https://github.com/TropComplique/lda2vec-pytorch)
		- > `pytorch` implementation of Moody's lda2vec, a way of topic modeling using word embeddings.
The original paper: [Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec](https://arxiv.org/abs/1605.02019).
	- [lda2vec – flexible & interpretable NLP models](https://lda2vec.readthedocs.io/en/latest/?badge=latest)
		- > An overview of the lda2vec Python module can be found here.
	- https://nbviewer.jupyter.org/github/cemoody/lda2vec/blob/master/examples/twenty_newsgroups/lda2vec/lda2vec.ipynb
- [Bayesian topic modeling](https://tedunderwood.com/category/methodology/topic-modeling/bayesian-topic-modeling/)
- [LDA in Python – How to grid search best topic models?](https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/#10diagnosemodelperformancewithperplexityandloglikelihood)
	- > Python’s Scikit Learn provides a convenient interface for topic modeling using algorithms like Latent Dirichlet allocation(LDA), LSI and Non-Negative Matrix Factorization. In this tutorial, you will learn how to build the best possible LDA topic model and explore how to showcase the outputs as meaningful results.
- Topic Modeling with BERT
	- > Topic modelling on the other hand focuses on categorising texts into particular topics. For this task, it is arguably arbitrary to use a language model since topic modelling focuses more on categorisation of texts, rather than the fluency of those texts. 
Thinking about it though, as well as the suggest given above, you could also develop separate language models, where for example, one is trained on texts within topic A, another in topic B, etc. Then you could categorise texts by outputting a probability distribution over topics. 
So, in this case, you might be able to do to transfer learning, whereby you take the pre-trained BERT model, add any additional layers, including a final output softmax layer, which produces the probability distribution over topics. To re-train the model, you essentially freeze the parameters within the BERT model itself and only train the additional layers you added
	- [Topic Modeling with BERT](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6)
	- [BERTopic](https://github.com/MaartenGr/BERTopic)
		- > BERTopic is a topic modeling technique that leverages 🤗 transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.
	- 
- [Short Text Topic Modeling](https://towardsdatascience.com/short-text-topic-modeling-70e50a57c883)
	- Gibbs Sampling Dirichlet Mixture Model (GSDMM)
		- [GSDMM: Short text clustering](https://github.com/rwalk/gsdmm)
	- [A dirichlet multinomial mixture model-based approach for short text clustering](https://dl.acm.org/doi/10.1145/2623330.2623715)
	- [Short Text Topic Modeling Techniques, Applications, and Performance: A Survey](https://arxiv.org/pdf/1904.07695.pdf)
- [Probabilistic Topic Models: Expectation-Maximization Algorithm](https://www.coursera.org/lecture/text-mining/3-5-probabilistic-topic-models-expectation-maximization-algorithm-part-2-naLsv)
- [kNN](https://stackoverflow.com/questions/16782114/topic-modelling-but-with-known-topics)


## Tools

- [pyLDAvis](https://github.com/bmabey/pyLDAvis)
	- > Python library for interactive topic model visualization.
- 
