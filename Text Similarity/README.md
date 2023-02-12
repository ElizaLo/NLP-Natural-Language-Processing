# Text Similarity 

# Understanding Similarity

Similarity is the distance between two vectors where the vector dimensions represent the features of two objects. In simple terms, similarity is the measure of how different or alike two data objects are. If the distance is small, the objects are said to have a high degree of similarity and vice versa. Generally, it is measured in the range 0 to 1. This score in the range of [0, 1] is called the similarity score.

An important point to remember about similarity is that it’s subjective and highly dependent on the domain and use case.

# 💠 Similarity Measures

## 🔹 Jaccard Index

Jaccard index, also known as Jaccard similarity coefficient, treats the data objects like sets. It is defined as the size of the intersection of two sets divided by the size of the union. Let’s continue with our previous example:

- **Sentence 1:** `The bottle is empty.`
- **Sentence 2:** `There is nothing in the bottle.`

To calculate the similarity using Jaccard similarity, we will first perform text normalization to reduce words their roots/lemmas. There are no words to reduce in the case of our example sentences, so we can move on to the next part. Drawing a Venn diagram of the sentences we get:

<img src="" width="1050" height="150"/>

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

<img src="" width="1050" height="150"/>

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
