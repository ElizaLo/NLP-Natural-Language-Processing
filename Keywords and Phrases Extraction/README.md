<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/img/Keywords_and_Phrases_Extraction.png" width="1050" height="150"/>

# 💠 Models

| Title | Description, Information |
| :---:         |          :--- |
| **GPT-3** | 📄 **Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165v4), [News Summarization and Evaluation in the Era of GPT-3](https://arxiv.org/pdf/2209.12356.pdf), [Papers with Code - GPT-3 Explained](https://paperswithcode.com/method/gpt-3)|

## 🔹 GPT-3: Generative Pre-trained Transformer

> Read more about GPT-3 model in 📂 [Language Models](https://github.com/ElizaLo/NLP-Natural-Language-Processing/tree/master/Language%20Models) folder

- 📄 **Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
  - 📃 Other related papers: [News Summarization and Evaluation in the Era of GPT-3](https://arxiv.org/abs/2209.12356)
  - [Papers with Code - GPT-3 Explained](https://paperswithcode.com/method/gpt-3)
- :hammer_and_wrench: **Implementations:**
  - [OpenAI API - GPT-3 Documentation](https://beta.openai.com/docs/models/gpt-3)
  - [Fine-tuning](https://beta.openai.com/docs/guides/fine-tuning)
- 📰 **Articles:**
  - [State of the Art GPT-3 Summarizer For Any Size Document or Format](https://www.width.ai/post/gpt3-summarizer)
- :gear: **Notebook:** 
  - [GPT-3.ipynb](https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/Text%20Summarization/GPT-3.ipynb) - GPT-3 - Generative Pre-trained Transformer 3, **_model_**: `text-davinci-003` (released: _November 2022_) 

### GPT-3 Tokenization

- [GPT-3 Online Tokenizer](https://beta.openai.com/tokenizer)

Tokenizer for GPT-3 is the same as GPT-2: 🤗 [OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#gpt2tokenizerfast).

A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).

The GPT-3 model can take as input from 4,000 to 2,000 tokens (**not confused with words!**). GPT-3 generates ~125-140% tokenks from the input text). 
> The text with 2,000 words approximately has 2,800 tokens.

Models understand and process text by breaking it down into tokens. Tokens can be words or just chunks of characters. For example, the word _**“hamburger”**_ gets broken up into the tokens _**“ham”**_, _**“bur”**_ and _**“ger”**_, while a short and common word like _**“pear”**_ is a single token. Many tokens start with a whitespace, for example _**“ hello”**_ and _**“ bye”**_.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/tokens.png" width="726" height="96"/>

> Common words like _**“cat”**_ are a single token, while less common words are often broken down into multiple tokens. _For example,_ _**“Butterscotch”**_ translates to four tokens: _**“But”**_, _**“ters”**_, _**“cot”**_, and _**“ch”**_.

The number of tokens processed in a given API request depends on the length of both your inputs and outputs. As a rough rule of thumb, 1 token is approximately 4 characters or 0.75 words for English text. One limitation to keep in mind is that your text prompt and generated completion combined must be no more than the model's maximum context length (for most models this is 2048 tokens, or about 1500 words). Check out [tokenizer tool](https://beta.openai.com/tokenizer) to learn more about how text translates to tokens.

### `text-davinci-003` model

> OpenAI releases a new language model for GPT-3 trained with human feedback (**November 2022**). It brings numerous improvements, according to OpenAI.

The new GPT-3 model `“text-davinci-003”` is based on the [InstructGPT](https://the-decoder.com/openai-million-dollar-investment-and-a-new-ai-model/) models introduced by OpenAI earlier this year, which are optimized with human feedback. These models have already shown that AI models trained with RLHF (Reinforcement Learning from Human Feedback) can achieve better results with the same or even lower parameters.

According to OpenAI alignment researcher Jan Leike, `“text-davinci-003”` is largely equivalent to the InstructGPT models, but is not identical. The new model “scores higher on human preference ratings without being fundamentally more capable” than the underlying base model. For fine-tuning, OpenAI required “very little compute and data to align it compared to pretraining”.

Leike points out that the new GPT model still has “important limitations” and, for example, sometimes simply makes up things. However, such missteps should now “hopefully” be less frequent and less serious.

`“text-davinci-003”` can generate **“clearer, more engaging, and more compelling content”** and handle **more complex instructions**, according to OpenAI.

`“text-davinci-003”` can also write longer texts, according to OpenAI. As a result, the language AI can now take on tasks that were previously unfeasible. 

📰 **Articles:**

- [OpenAI’s latest GPT-3 model generates better and longer texts](https://the-decoder.com/openais-latest-gpt-3-model-generates-better-and-longer-texts/)

###  ⚙️ Fine-tuning model

> Read all details about **Fine-tuning GPT-3 model** in 📂 [Language Models - ⚙️ Fine-tuning](https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/Language%20Models/README.md#%EF%B8%8F-fine-tuning-model) folder

Fine-tuning lets you get more out of the models available through the API by providing:

1. Higher quality results than prompt design
2. Ability to train on more examples than can fit in a prompt
3. Token savings due to shorter prompts
4. Lower latency requests

- 🔸 **Change prompt**

From → `Extract keywords from this text:`, to → `Extract keywords and key phrases from this text:`

💠 **Prompt Design**

This simple text-in, text-out interface means you can "program" the model by providing instructions or just a few examples of what you'd like it to do. Its success generally depends on the complexity of the task and the quality of your prompt. A good rule of thumb is to think about how you would write a word problem for a middle schooler to solve. A well-written prompt provides enough information for the model to know what you want and how it should respond.

GPT-3 models by OpenAI can do everything from generating original stories to performing complex text analysis. Because they can do so many things, you have to be explicit in describing what you want. Showing, not just telling, is often the secret to a good prompt.

**There are basic guidelines to creating prompts:**

- **Show and tell.** Make it clear what you want either through instructions, examples, or a combination of the two. If you want the model to rank a list of items in alphabetical order or to classify a paragraph by sentiment, show it that's what you want. 
- **Use plain language to describe your inputs and outputs.** As a best practice, start with plain language descriptions. While you can often use shorthand or keys to indicate the input and output, it's best to start by being as descriptive as possible and then working backwards to remove extra words and see if performance stays consistent
- **Provide quality data.** If you're trying to build a classifier or get the model to follow a pattern, make sure that there are enough examples. Be sure to proofread your examples — the model is usually smart enough to see through basic spelling mistakes and give you a response, but it also might assume this is intentional and it can affect the response. → that’s why you may need text pre-processing/cleaning and Grammatical Error Correction (GEC).
- **You need fewer examples for familiar tasks.** For Tweet sentiment classifier, you don't need to provide any examples. This is because the API already has an understanding of sentiment and the concept of a Tweet. If you're building a classifier for something the API might not be familiar with, it might be necessary to provide more examples.
- **Check your settings.** The `temperature` and `top_p` settings control how deterministic the model is in generating a response. If you're asking it for a response where there's only one right answer, then you'd want to set these lower. If you're looking for more diverse responses, then you might want to set them higher. The number one mistake people use with these settings is assuming that they're "cleverness" or "creativity" controls.

‼️ **Troubleshooting**

If you're having trouble getting the API to perform as expected, follow this checklist:
1. Is it clear what the intended generation should be?
2. Are there enough examples?
3. Did you check your examples for mistakes? (The API won't tell you directly)
4. Are you using `temperature` and `top_p` correctly?

💠 **Separator**

As a separator we can use `\nKeywords and Phrases:` which clearly separated the prompt from the completion. With a sufficient number of examples, the separator doesn't make much of a difference (usually less than 0.4%) as long as it doesn't appear within the prompt or the completion.

💠 **Conditional generation**

Conditional generation is a problem where the content needs to be generated given some kind of input. This includes paraphrasing, summarizing, entity extraction, product description writing given specifications, chatbots and many others. For this type of problem we recommend:

- Use a separator at the end of the prompt, e.g. `\n\n###\n\n`. Remember to also append this separator when you eventually make requests to your model.
- Use an ending token at the end of the completion, _e.g._ `END`
- Remember to add the ending token as a stop sequence during inference, _e.g._ `stop=[" END"]`
- Aim for at least ~500 examples
- Ensure that the prompt + completion doesn't exceed 2048 tokens, including the separator
- Ensure the examples are of high quality and follow the same desired format
- Ensure that the dataset used for finetuning is very similar in structure and type of task as what the model will be used for
- Using Lower learning rate and only 1-2 epochs tends to work better for these use cases

▶️ **_Case study: Write an engaging ad based on a Wikipedia article_**

This is a generative use case so you would want to ensure that the samples you provide are of the highest quality, as the fine-tuned model will try to imitate the style (and mistakes) of the given examples. A good starting point is around 500 examples. A sample dataset might look like this:

```
{"prompt":"<Product Name>\n<Wikipedia description>\n\n###\n\n", "completion":" <engaging ad> END"}
```

_For example:_

```
{"prompt":"Samsung Galaxy Feel\nThe Samsung Galaxy Feel is an Android smartphone developed by Samsung Electronics exclusively for the Japanese market. The phone was released in June 2017 and was sold by NTT Docomo. It runs on Android 7.0 (Nougat), has a 4.7 inch display, and a 3000 mAh battery.\nSoftware\nSamsung Galaxy Feel runs on Android 7.0 (Nougat), but can be later updated to Android 8.0 (Oreo).\nHardware\nSamsung Galaxy Feel has a 4.7 inch Super AMOLED HD display, 16 MP back facing and 5 MP front facing cameras. It has a 3000 mAh battery, a 1.6 GHz Octa-Core ARM Cortex-A53 CPU, and an ARM Mali-T830 MP1 700 MHz GPU. It comes with 32GB of internal storage, expandable to 256GB via microSD. Aside from its software and hardware specifications, Samsung also introduced a unique a hole in the phone's shell to accommodate the Japanese perceived penchant for personalizing their mobile phones. The Galaxy Feel's battery was also touted as a major selling point since the market favors handsets with longer battery life. The device is also waterproof and supports 1seg digital broadcasts using an antenna that is sold separately.\n\n###\n\n", "completion":"Looking for a smartphone that can do it all? Look no further than Samsung Galaxy Feel! With a slim and sleek design, our latest smartphone features high-quality picture and video capabilities, as well as an award winning battery life. END"}
```

Here we used a multi line separator, as Wikipedia articles contain multiple paragraphs and headings. We also used a simple end token, to ensure that the model knows when the completion should finish.

▶️ **_Case study: Entity extraction_**

This is similar to a language transformation task. To improve the performance, it is best to either sort different extracted entities alphabetically or in the same order as they appear in the original text. This will help the model to keep track of all the entities which need to be generated in order. The dataset could look as follows:

```
{"prompt":"<any text, for example news article>\n\n###\n\n", "completion":" <list of entities, separated by a newline> END"}
```

_For example:_

```
{"prompt":"Portugal will be removed from the UK's green travel list from Tuesday, amid rising coronavirus cases and concern over a \"Nepal mutation of the so-called Indian variant\". It will join the amber list, meaning holidaymakers should not visit and returnees must isolate for 10 days...\n\n###\n\n", "completion":" Portugal\nUK\nNepal mutation\nIndian variant END"}
```

A multi-line separator works best, as the text will likely contain multiple lines. Ideally there will be a high diversity of the types of input prompts (news articles, Wikipedia pages, tweets, legal documents), which reflect the likely texts which will be encountered when extracting entities.

- 🔸 **Temperature**

> See more details and examples in 📂 [Language Models](https://github.com/ElizaLo/NLP-Natural-Language-Processing/tree/master/Language%20Models#fine-tuning-model) folder

Remember that the model predicts which text is most likely to follow the text preceding it. Temperature is a value between 0 and 1 that essentially lets you control how confident the model should be when making these predictions. Lowering temperature means it will take fewer risks, and completions will be more accurate and deterministic. Increasing temperature will result in more diverse completions.

It’s usually best to set a low temperature for tasks where the desired output is well-defined. Higher temperature may be useful for tasks where variety or creativity are desired, or if you'd like to generate a few variations for your end users or human experts to choose from.

‼️ The actual completion you see may differ because the API is stochastic by default. This means that you might get a slightly different completion every time you call it, even if your prompt stays the same. You can control this behavior with the [temperature](https://beta.openai.com/docs/api-reference/completions/create#completions/create-temperature) setting.

📰 **Articles**

- [How to sample from language models](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277)

### 💭 Conclusions

- The same `temperature` value generated different amounts of words and different words and phrases.
- This model is very sensitive to the input end, which influences the tokenization and consequently affects the extracted keywords.
    > **Warning:** Your text ends in a trailing space, which causes worse performance due to how the API splits text into tokens.
- It’s also worth taking note of the rather high `temperature` value and shorter max tokens. 
- Each time model generates text of different lengths. Setting up `max_tokens` doesn’t guarantee that text would be this length.
- Words in upper case are divided wrong into the tokens (e.g. FEBRUARY, CUISINE, RESEARCH, MENU, etc.). It is necessary to reduce all words to lowercase or to the form of ordinary sentences, where only the capital letter in the word is large.

# 🔺 Evaluation Metrics

- NDCG (Normalized Discounted Cumulative Gain) 
- MRR (Mean Reciprocal Rank) 
- F1
- MAP (Mean Average Precision)
- Extracting the percentage of shared words between two texts
- **_Text Similarity Evaluation Metrics for Keywords Extraction:_**
  - Jaccard Index
  - Euclidean Distance
  - Cosine Similarity
  - Word Mover’s Distance
- 

## 💠 Extracting the percentage of shared words between two texts

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

## 📃 Text Similarity Evaluation Metrics

Text similarity has to determine how ‘close’ two pieces of text are both in surface closeness [lexical similarity] and meaning [semantic similarity].

For instance, how similar are the phrases _**“the cat ate the mouse”**_ with _**“the mouse ate the cat food”**_ by just looking at the words?

- On the surface, if you consider only **word level similarity**, these two phrases appear very similar as 3 of the 4 unique words are an exact overlap. It typically does not take into account the actual meaning behind words or the entire phrase in context.
- Instead of doing a **word for word comparison**, we also need to pay attention to context in order to capture more of the semantics. To consider semantic similarity we need to focus on **phrase/paragraph levels (or lexical chain level)** where a piece of text is broken into a relevant group of related words prior to computing similarity. We know that while the words significantly overlap, these two phrases actually have **different meaning**.

### Understanding Similarity

Similarity is the distance between two vectors where the vector dimensions represent the features of two objects. In simple terms, similarity is the measure of how different or alike two data objects are. If the distance is small, the objects are said to have a high degree of similarity and vice versa. Generally, it is measured in the range 0 to 1. This score in the range of [0, 1] is called the similarity score.

An important point to remember about similarity is that it’s subjective and highly dependent on the domain and use case.

### 💠 Similarity Measures

#### 🔹 Jaccard Index

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

#### 🔹 Euclidean Distance

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

#### 🔹 Cosine Similarity

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

#### 🔹 Word Mover’s Distance

WMD uses the **word embeddings of the words in two texts to measure the minimum distance** that the words in one text need to “travel” in semantic space to reach the words in the other text.

**🛠️ Implementation**

- [Word Mover’s Distance](https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html), Gensim’s implemenation of the WMD

## ❓What Metric To Use?

Jaccard similarity takes into account only the set of unique words for each text document. This makes it the likely candidate for assessing the similarity of documents when repetition is not an issue. A prime example of such an application is comparing product descriptions. For instance, if a term like _“HD”_ or _“thermal efficiency”_ is used multiple times in one description and just once in another, the Euclidean distance and cosine similarity would drop. On the other hand, if the total number of unique words stays the same, the Jaccard similarity will remain unchanged. 

Both Euclidean and cosine similarity metrics drop if an additional _‘empty’_ is added to our first example sentence:

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Similarity/img/example.png" width="492" height="167"/>

Jaccard similarity is rarely used when working with text data as it does not work with text embeddings. This means that is limited to assessing the lexical similarity of text, i.e., how similar documents are on a word level.

As far as cosine and Euclidean metrics are concerned, the differentiating factor between the two is that cosine similarity is not affected by the magnitude/length of the feature vectors. Let’s say we are creating a topic tagging algorithm. If a word _(e.g. senate)_ occurs more frequently in document 1 than it does in document 2,  we could assume that document 1 is more related to the topic of _Politics_. However, it could also be the case that we are working with news articles of different lengths. Then, the word _‘senate’_ probably occurred more in document 1 simply because it was way longer. As we saw earlier when the word _‘empty’_ was repeated, cosine similarity is less sensitive to a difference in lengths.

In addition to that, Euclidean distance doesn’t work well with the sparse vectors of text embeddings. **So cosine similarity is generally preferred over Euclidean distance when working with text data.** The only length-sensitive text similarity use case that comes to mind is plagiarism detection. 

## 📚 Books

- [13 Measuring text similarities · Data Science Bookcamp: Five Python projects](https://livebook.manning.com/book/data-science-bookcamp/chapter-13/1)

## 📄 Papers

- [Measurement of Text Similarity: A Survey](https://www.mdpi.com/2078-2489/11/9/421)


## 📰 Articles

- [Ultimate Guide To Text Similarity With Python - NewsCatcher](https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python)
- [Text Similarity Measures](https://machinelearninggeek.com/text-similarity-measures/#Cosine_Similarity_using_Spacy)
- [Text Similarities : Estimate the degree of similarity between two texts](https://medium.com/@adriensieg/text-similarities-da019229c894)
    - :octocat:[adsieg/text_similarity](https://github.com/adsieg/text_similarity) 
- [Overview of Text Similarity Metrics in Python](https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50)
- [Cosine Similarity – Text Similarity Metric](https://studymachinelearning.com/cosine-similarity-text-similarity-metric/)
- [Understanding and Using Common Similarity Measures for Text Analysis](https://programminghistorian.org/en/lessons/common-similarity-measures#cosine-similarity-and-cosine-distance)


# 💡 Ideas

- Run the model with different `temperature` values and join results together to achieve more extracted keywords. 
- Run the model with different `temperature` values and count how many words are generated with each.
- Reduce all words to lowercase or to the form of ordinary sentences, where only the capital letter in the word is large.
- Clean texts to make them more structured and coherent.
- **Chunk/divide the input text:** divide the text into several smaller ones and make a keywords extraction for each of them separately and then combine it into one concatenated keywords list.
- Make chunking of documents, try and compare the results of summarization of **different GPT-3 models**: `text-davinci-003`, `text-curie-001`, `text-babbage-001` and `text-ada-001`.

# ❓ Questions

- How to evaluate results?
- How to choose `temperature` value?

