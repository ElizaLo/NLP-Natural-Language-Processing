# Topic Modeling

Topic modelling is an unsupervised machine learning method that **helps us discover hidden semantic structures in a paper**, that allows us to learn topic representations of papers in a corpus. The model can be applied to any kinds of labels on documents, such as tags on posts on the website.

> Ещё можно называть мягкой би-кластеризацией

> Dataset for this task:
  > - [A Million News Headlines](https://www.kaggle.com/therohk/million-headlines) - News headlines published over a period of 18 Years 
  
- [Topic Modeling with LSA, PLSA, LDA & lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)
- [Topic Modeling with Gensim (Python)](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#18dominanttopicineachsentence)
- [Topic Modeling in Python](https://ourcodingclub.github.io/tutorials/topic-modelling-python/)
- [Topic modeling visualization – How to present the results of LDA models?](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)

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


### Code examples

- [LDA_news_headlines](https://github.com/susanli2016/NLP-with-Python/blob/master/LDA_news_headlines.ipynb)


<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_15.png" width="606" height="502">


## Guided Latent Dirichlet Allocation (LDA)

`GuidedLDA` OR `SeededLDA` implements latent Dirichlet allocation (LDA) using collapsed Gibbs sampling. `GuidedLDA` can be guided by setting some seed words per topic. Which will make the topics converge in that direction.

- [Incorporating Lexical Priors into Topic Models](https://www.aclweb.org/anthology/E12-1021.pdf) Paper
- [GuidedLDA’s documentation](https://guidedlda.readthedocs.io/en/latest/)
- [GuidedLDA](https://github.com/vi3k6i5/guidedlda) GitHub repository
- [How we Changed Unsupervised LDA to Semi-Supervised GuidedLDA](https://www.freecodecamp.org/news/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164/) Article

### Articles 

- [How did I tackle a real-world problem with GuidedLDA?](https://medium.com/analytics-vidhya/how-i-tackled-a-real-world-problem-with-guidedlda-55ee803a6f0d)
  - [Guided LDA_6topics-4-grams](https://github.com/ShahrzadH/Insight_Project_SHV/blob/master/notebook/Guided%20LDA_6topics-4-grams.ipynb) Code Exapmle
- 

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



## Regulariazation

### Kullback–Leibler divergence (relative entropy)

Способ померять растояние между двумя распределениями

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_13.png" width="618" height="456">
<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/img/formula_14.png" width="613" height="189">
