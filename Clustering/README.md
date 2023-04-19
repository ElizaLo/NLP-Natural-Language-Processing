<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/img/Text_Clustering.png" width="1050" height="150"/>

# 💠 Making clustering based on Word Embeddings

## 🔹 GPT word embeddings by OpenAI

### ⚙️ Example notebooks

- [Obtain_dataset.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/Obtain_dataset.ipynb)
- [Clustering.ipynb](https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb)

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

# Conclusions

- It's important to note that clusters will not necessarily match what you intend to use them for. **A larger amount of clusters will focus on more specific patterns, whereas a small number of clusters will usually focus on largest discrepencies in the data.**

# 🛠️ Libraries, frameworks, etc.

| Title | Description, Information |
| :---:         |          :--- |
|[PyCaret](https://pycaret.gitbook.io/docs/get-started/quickstart#clustering)|The goal is to predict the categorical class labels which are discrete and unordered. Some common use cases include predicting customer default (Yes or No), predicting customer churn (customer will leave or stay), the disease found (positive or negative).|
