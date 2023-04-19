<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/img/Text_Clustering.png" width="1050" height="150"/>

# 💠 Making clustering based on Word Embeddings

## 🔹 GPT word embeddings by OpenAI

- 🛠️ **Implementation:**
    - [Embeddings - OpenAI API](https://platform.openai.com/docs/guides/embeddings)
    - [Embeddings - API Reference - OpenAI API](https://platform.openai.com/docs/api-reference/embeddings)
- ⚙️ **Example notebooks:** 
    - [Obtain_dataset.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb)
    - [Clustering.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb)

```python
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    
# This may take a few minutes
df["Text"] = df["Text"].apply(lambda x: get_embedding(x, model=embedding_model))
```

# 💠 K-Means

- 🛠️ **Implementation:** [sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

## 💭 Conclusions

- The bigger amount of clusters the more “accurate” and defined, meaningful and topic-related clusterisation _(e.g. 400-800 clusters)_.
- If we have a phrase and not a single word _(e.g. Sports card store, Sheet music store, Used book store, etc.)_ the last word weight the most for this clusterisation algorithm _(e.g. in this case it would be the word “store”, in other examples it was “restaurant”, “bar”, “clinic”, “consultant”, etc.)_.

# 💠 Find the optimal amount of clusters

```python
Sum_of_squared_distances = []
K = range(100,800, 50)

for k in K:
    print(k)
    km = KMeans(n_clusters=k, max_iter=300, n_init=10)
    km = km.fit(matrix)
    Sum_of_squared_distances.append(km.inertia_)

    
plt.figure(figsize=(16,8))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
```

# 💭 Conclusions

- It's important to note that clusters will not necessarily match what you intend to use them for. **A larger amount of clusters will focus on more specific patterns, whereas a small number of clusters will usually focus on largest discrepencies in the data.**

# 🛠️ Libraries, frameworks, etc.

| Title | Description, Information |
| :---:         |          :--- |
|[sklearn.cluster](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) module|The sklearn.cluster module gathers popular unsupervised clustering algorithms.|
|[PyCaret](https://pycaret.gitbook.io/docs/get-started/quickstart#clustering)|The goal is to predict the categorical class labels which are discrete and unordered. Some common use cases include predicting customer default (Yes or No), predicting customer churn (customer will leave or stay), the disease found (positive or negative).|
