# All Word Embedings

## Bag of Words(BoW)

### Count Vectorizer

CountVectorizer tokenizes(tokenization means dividing the sentences in words) the text along with performing very basic preprocessing. It removes the punctuation marks and converts all the words to lowercase.
The vocabulary of known words is formed which is also used for encoding unseen text later.
An encoded vector is returned with a length of the entire vocabulary and an integer count for the number of times each word appeared in the document. The image below shows what I mean by the encoded vector.

> Count Vectorizer sparse matrix representation of words. (a) is how you visually think about it. (b) is how it is really represented in practice.

The row of the above matrix represents the document, and the columns contain all the unique words with their frequency. In case a word did not occur, then it is assigned zero correspondings to the document in a row.
Imagine it as a one-hot encoded vector and due to that, it is pretty obvious to get a sparse matrix with a lot of zeros.

- [10+ Examples for Using CountVectorizer](https://kavita-ganesan.com/how-to-use-countvectorizer/#.YGMSUy1c5WO)
