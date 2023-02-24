<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/img/Text_Similarity.png" width="1050" height="150"/>

Text similarity has to determine how ‘close’ two pieces of text are both in surface closeness [lexical similarity] and meaning [semantic similarity].

For instance, how similar are the phrases _**“the cat ate the mouse”**_ with _**“the mouse ate the cat food”**_ by just looking at the words?

- On the surface, if you consider only **word level similarity**, these two phrases appear very similar as 3 of the 4 unique words are an exact overlap. It typically does not take into account the actual meaning behind words or the entire phrase in context.
- Instead of doing a **word for word comparison**, we also need to pay attention to context in order to capture more of the semantics. To consider semantic similarity we need to focus on **phrase/paragraph levels (or lexical chain level)** where a piece of text is broken into a relevant group of related words prior to computing similarity. We know that while the words significantly overlap, these two phrases actually have **different meaning**.

# Understanding Similarity

Similarity is the distance between two vectors where the vector dimensions represent the features of two objects. In simple terms, similarity is the measure of how different or alike two data objects are. If the distance is small, the objects are said to have a high degree of similarity and vice versa. Generally, it is measured in the range 0 to 1. This score in the range of [0, 1] is called the similarity score.

An important point to remember about similarity is that it’s subjective and highly dependent on the domain and use case.

# 💠 Extracting the percentage of shared words between two texts

```python
def percent_shared(s1, s2):
    
    s_1 = set(s1.split())
    s_2 = set(s2.split())

    shared_words = s_1 & s_2
    diverging_words = s_1 ^ s_2
    total_words = s_1 | s_2
    assert len(total_words) == len(shared_words) + len(diverging_words)
    
    percent_shared = 100 * len(shared_words) / len(total_words)
    percent_diverging = 100 * len(diverging_words) / len(total_words)
    
    percent_to_s1 = round(len(shared_words) * 100 / len(s_1), 2)
    percent_to_s2 = round(len(shared_words) * 100 / len(s_2), 2)
    
    print(f"Together, extracted and correct keywords contain {len(total_words)} "
          f"unique words. \n{percent_shared:.2f}% of these words are "
          f"shared. \n{percent_diverging:.2f}% of these words diverge.\n")

    print("Overlap words -- ", len(shared_words))
    print('Percent to s1 keywords -- ', percent_to_s1, "%")
    print('Percent to s2 keywords -- ', percent_to_s2, "%")
```

# 💠 Similarity Measures

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/text_similarity_measures.png" width="765" height="662"/>

## 🔹 Jaccard Index

Jaccard index, also known as Jaccard similarity coefficient, treats the data objects like sets. It is defined as the size of the intersection of two sets divided by the size of the union. Let’s continue with our previous example:

- **Sentence 1:** `The bottle is empty.`
- **Sentence 2:** `There is nothing in the bottle.`

To calculate the similarity using Jaccard similarity, we will first perform text normalization to reduce words their roots/lemmas. There are no words to reduce in the case of our example sentences, so we can move on to the next part. Drawing a Venn diagram of the sentences we get:

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/Jaccard_Index.png" width="375" height="223"/>

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/Jaccard_Index_formula.png" width="400" height="108"/>

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/Jaccard_Index_formula_2.png" width="317" height="159"/>

- _Size of the intersection of the two sets:_ **3**
- _Size of the union of the two sets:_ **1+3+3 = 7**
- _Using the Jaccard index, we get a similarity score of_ **3/7 = 0.42**

```python
def jaccard_similarity(x,y):
  """ returns the jaccard similarity between two lists """
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)
```

## 🔹 Euclidean Distance

Euclidean distance, or L2 norm, is the most commonly used form of the [Minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance). Generally speaking, when people talk about distance, they refer to Euclidean distance. It uses the Pythagoras theorem to calculate the distance between two points as indicated in the figure below:

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/Pythagoras_theorem.png" width="375" height="225"/>

The larger the distance d between two vectors, the lower the similarity score and vice versa. 

Let’s compute the similarity between our example statements using Euclidean distance:

```python
from math import sqrt, pow, exp
 
def squared_sum(x):
  """ return 3 rounded square rooted value """
 
  return round(sqrt(sum([a*a for a in x])),3)
 
def euclidean_distance(x,y):
  """ return euclidean distance between two lists """
 
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
```

To compute the Euclidean distance we need vectors, so we’ll use spaCy’s in-built Word2Vec model to create text embeddings. 

```python
embeddings = [nlp(sentence).vector for sentence in sentences]

distance = euclidean_distance(embeddings[0], embeddings[1])
print(distance)

# OUTPUT
1.8646982721454675
```

Okay, so we have the Euclidean distance of 1.86, but what does that mean? See, the problem with using distance is that it’s hard to make sense if there is nothing to compare to. The distances can vary from 0 to infinity, we need to use some way to normalize them to the range of 0 to 1.

Although we have our typical normalization formula that uses mean and standard deviation, it is sensitive to outliers. That means if there are a few extremely large distances, every other distance will become smaller as a consequence of the normalization operation. So the best option here is to use something like the Euler’s constant as follows: **1/e<sup>d</sup>**

```python

def distance_to_similarity(distance):
  return 1/exp(distance)

distance_to_similarity(distance) 

# OUTPUT
0.8450570465624478
```

## 🔹 Cosine Similarity

Cosine Similarity computes the similarity of two vectors as the cosine of the angle between two vectors. It determines whether two vectors are pointing in roughly the same direction. So if the angle between the vectors is 0 degrees, then the cosine similarity is 1. 

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/Cosine_Similarity.png" width="375" height="189"/>

It is given as:

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/Cosine_Similarity_formula.png" width="356" height="175"/>

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/Cosine_Similarity_formula_2.png" width="533" height="139"/>

Where **||v||** represents the length of the vector **v**, **𝜃** denotes the angle between **v** and **w**, and **‘.’** denotes the dot product operator.

```python
def cos_similarity(x,y):

  '''return cosine similarity between two lists'''
 
  numerator = sum(a*b for a,b in zip(x,y))
  denominator = squared_sum(x)*squared_sum(y)
  return round(numerator/float(denominator),3)

cos_similarity(embeddings[0], embeddings[1])

# OUTPUT
0.891
```

- The implementation of Cosine Similarity in Python using TF-IDF vector of Scikit-learn:

```python
# Let's import text feature extraction TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Cosien Similarity metric
from sklearn.metrics.pairwise import cosine_similarity


docs=['I like dogs.', 'I hate dogs.']

# Create TFidfVectorizer 
tfidf= TfidfVectorizer()

# Fit and transform the documents 
tfidf_vector = tfidf.fit_transform(docs)

# Compute cosine similarity
cosine_sim=cosine_similarity(tfidf_vector, tfidf_vector)

# Print the cosine similarity
print(cosine_sim)
```

- Cosine Similarity using **Spacy**

We can also use cosine using [Spacy](https://machinelearninggeek.com/text-analytics-for-beginners-using-python-spacy-part-1/) similarity method:

```python
import en_core_web_sm

nlp = en_core_web_sm.load()

## try medium or large spacy english models

doc1 = nlp("I like apples.")

doc2 = nlp("I like oranges.")

# cosine similarity
doc1.similarity(doc2) 
```

- Cosine Similarity using **Scipy**

We can also implement using Scipy:

```python
from scipy import spatial

# Document Vectorization
doc1, doc2 = nlp('I like apples.').vector, nlp('I like oranges.').vector

# Cosine Similarity
result = 1 - spatial.distance.cosine(doc1, doc2)

print(result)
```

```python
# Cosine Similarity - Spacy
import spacy
import en_core_web_sm

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

def cosine_similarity_spacy(s1, s2):
    s_1 = nlp(s1)
    s_2 = nlp(s2)
    return s_1.similarity(s_2)


# Cosine Similarity - Scipy
from scipy import spatial

def cosine_similarity_scipy(s1, s2):
    # Document Vectorization
    s_1 = nlp(s1).vector
    s_2 = nlp(s2).vector
    return 1 - spatial.distance.cosine(s_1, s_2)
    
# CountVectorizer Method + Cosine Similarity
def cosine_distance_countvectorizer_method(s1, s2):
    
    # sentences to list
    allsentences = [s1 , s2]
    
    # packages
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy.spatial import distance
    
    # text to vector
    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(allsentences)
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
    
    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    print('Similarity of two sentences are equal to ',round((1-cosine)*100,2),'%')
    return cosine
```

Cosine similarity is best suitable for where repeated words are more important and can work on any size of the document.

## 🔹 Word Mover’s Distance

WMD uses the **word embeddings of the words in two texts to measure the minimum distance** that the words in one text need to “travel” in semantic space to reach the words in the other text.

### 🛠️ Implementation

- [Word Mover’s Distance](https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html), Gensim’s implemenation of the WMD

## 🔹 Tanimoto Coefficient Similarity Method

### 📄 Papers

- [Tanimoto coefficient based Word Sense Disambiguation](https://www.ijarnd.com/manuscripts/v4i7/V4I7-1141.pdf)
- [Efficient identification of Tanimoto nearest neighbors](https://link.springer.com/article/10.1007/s41060-017-0064-z)

## 🔹 Semantic Textual Similarity

Semantic Textual Similarity (STS) assigns a score on the similarity of two texts.

### 📄 Papers

- [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175), 2018
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf), 2019

### 📰 Articles

- [BERT For Measuring Text Similarity](https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1)

### 🛠️ Implementation

- [Semantic Textual Similarity by SBERT.net](https://www.sbert.net/examples/training/sts/README.html)

# ❓What Metric To Use?

Jaccard similarity takes into account only the set of unique words for each text document. This makes it the likely candidate for assessing the similarity of documents when repetition is not an issue. A prime example of such an application is comparing product descriptions. For instance, if a term like _“HD”_ or _“thermal efficiency”_ is used multiple times in one description and just once in another, the Euclidean distance and cosine similarity would drop. On the other hand, if the total number of unique words stays the same, the Jaccard similarity will remain unchanged. 

Both Euclidean and cosine similarity metrics drop if an additional _‘empty’_ is added to our first example sentence:

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/example.png" width="492" height="167"/>

Jaccard similarity is rarely used when working with text data as it does not work with text embeddings. This means that is limited to assessing the lexical similarity of text, i.e., how similar documents are on a word level.

As far as cosine and Euclidean metrics are concerned, the differentiating factor between the two is that cosine similarity is not affected by the magnitude/length of the feature vectors. Let’s say we are creating a topic tagging algorithm. If a word _(e.g. senate)_ occurs more frequently in document 1 than it does in document 2,  we could assume that document 1 is more related to the topic of _Politics_. However, it could also be the case that we are working with news articles of different lengths. Then, the word _‘senate’_ probably occurred more in document 1 simply because it was way longer. As we saw earlier when the word _‘empty’_ was repeated, cosine similarity is less sensitive to a difference in lengths.

In addition to that, Euclidean distance doesn’t work well with the sparse vectors of text embeddings. **So cosine similarity is generally preferred over Euclidean distance when working with text data.** The only length-sensitive text similarity use case that comes to mind is plagiarism detection. 


# 📚 Books

- [13 Measuring text similarities · Data Science Bookcamp: Five Python projects](https://livebook.manning.com/book/data-science-bookcamp/chapter-13/1)

# 📄 Papers

- [Measurement of Text Similarity: A Survey](https://www.mdpi.com/2078-2489/11/9/421)


# 📰 Articles

- [Ultimate Guide To Text Similarity With Python - NewsCatcher](https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python)
- [Text Similarity Measures](https://machinelearninggeek.com/text-similarity-measures/#Cosine_Similarity_using_Spacy)
- [Text Similarities : Estimate the degree of similarity between two texts](https://medium.com/@adriensieg/text-similarities-da019229c894)
    - :octocat:[adsieg/text_similarity](https://github.com/adsieg/text_similarity) 
- [Overview of Text Similarity Metrics in Python](https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50)
- [Cosine Similarity – Text Similarity Metric](https://studymachinelearning.com/cosine-similarity-text-similarity-metric/)
- [Understanding and Using Common Similarity Measures for Text Analysis](https://programminghistorian.org/en/lessons/common-similarity-measures#cosine-similarity-and-cosine-distance)
- [What is Text Similarity and How to Implement it?](https://medium.com/msackiit/what-is-text-similarity-and-how-to-implement-it-c74c8b641883)
