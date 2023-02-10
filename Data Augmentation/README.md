<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/img/Data_Augmentation_img.png" width="1050" height="150"/>

[**Data augmentation**](https://en.wikipedia.org/wiki/Data_augmentation) in data analysis are techniques used to increase the amount of data by adding slightly modified copies of already existing data or newly created synthetic data from existing data. It acts as a [regularizer](https://en.wikipedia.org/wiki/Regularization_(mathematics)) and helps reduce [overfitting](https://en.wikipedia.org/wiki/Overfitting) when training a machine learning model. It is closely related to [oversampling](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis) in data analysis.

**Data Augmentation techniques are applied on below three levels:**

- Character Level
- Word Level
- Phrase Level
- Document Level

# 💠 Lexical Substitution

These approaches try to substitute words present in a text without changing the meaning of the sentence.

## 🔹 Thesaurus-based substitution

In this technique, we take a random word from the sentence and replace it with its synonym using a Thesaurus. _For example,_ we could use the [WordNet](https://wordnet.princeton.edu/) database for English to look up the synonyms and then perform the replacement. It is a manually curated database with relations between words.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-wordnet.png" width="589" height="193"/>

[Zhang et al.](https://arxiv.org/abs/1509.01626) used this technique in their 2015 paper “Character-level Convolutional Networks for Text Classification”. [Mueller et al.](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023) used a similar strategy to generate additional 10K training examples for their sentence similarity model. This technique was also used by [Wei et al.](https://arxiv.org/abs/1901.11196) as one of the techniques in their pool of four random augmentations in the “Easy Data Augmentation” paper.

For implementation, NLTK provides a programmatic [access to WordNet](https://www.nltk.org/howto/wordnet.html). You can also use the [TextBlob API](https://textblob.readthedocs.io/en/dev/quickstart.html#wordnet-integration). Additionally, there is a database called [PPDB](http://paraphrase.org/#/download) containing millions of paraphrases that you can download and use programmatically.

## 🔹 Synonym Replacement

Randomly choose _n_ words from the sentence that are not stop words. Replace each of these words with one of its synonyms chosen at random. 

_For example,_ given the sentence:

_This **article** will focus on summarizing data augmentation **techniques** in NLP._

The method randomly selects n words (say two), the words _**article**_ and _**techniques**_, and replaces them with _**write-up**_ and _**methods**_ respectively.

_This **write-up** will focus on summarizing data augmentation **methods** in NLP._

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)|This article offers an empirical exploration on the use of character-level convolutional networks (ConvNets) for text classification. We constructed several large-scale datasets to show that character-level convolutional networks could achieve state-of-the-art or competitive results. Comparisons are offered against traditional models such as bag of words, n-grams and their TFIDF variants, and deep learning models such as word-based ConvNets and recurrent neural networks.|
|[Siamese Recurrent Architectures for Learning Sentence Similarity](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12195/12023)|We present a siamese adaptation of the Long Short-Term Memory (LSTM) network for labeled data comprised of pairs of variable-length sequences. Our model is applied to as- sess semantic similarity between sentences, where we ex- ceed state of the art, outperforming carefully handcrafted features and recently proposed neural network systems of greater complexity. For these applications, we provide word- embedding vectors supplemented with synonymic informa- tion to the LSTMs, which use a fixed size vector to encode the underlying meaning expressed in a sentence (irrespective of the particular wording/syntax). By restricting subsequent operations to rely on a simple Manhattan metric, we compel the sentence representations learned by our model to form a highly structured space whose geometry reflects complex semantic relationships. Our results are the latest in a line of findings that showcase LSTMs as powerful language models capable of tasks requiring intricate understanding.|
|[EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)|<p>We present EDA: easy data augmentation techniques for boosting performance on text classification tasks. EDA consists of four simple but powerful operations: synonym replacement, random insertion, random swap, and random deletion. On five text classification tasks, we show that EDA improves performance for both convolutional and recurrent neural networks. EDA demonstrates particularly strong results for smaller datasets; on average, across five datasets, training with EDA while using only 50% of the available training set achieved the same accuracy as normal training with all available data. We also performed extensive ablation studies and suggest parameters for practical use.</p><ul><li> 📰 **Article** [These are the Easiest Data Augmentation Techniques in Natural Language Processing you can think of — and they work.](https://towardsdatascience.com/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610)</li><li> :octocat: [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://github.com/jasonwei20/eda_nlp)</li></ul>|

###  ⚙️ Tools

| Title | Description, Information |
| :---:         |          :--- |
|[NLTK :: Sample usage for wordnet](https://www.nltk.org/howto/wordnet.html)|NLTK provides a programmatic access to WordNet|
|[WordNet](https://wordnet.princeton.edu)|WordNet is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations. The resulting network of meaningfully related words and concepts can be navigated with the [browser](http://wordnetweb.princeton.edu/perl/webwn). WordNet is also freely and publicly available for [download](https://wordnet.princeton.edu/node/5). WordNet's structure makes it a useful tool for computational linguistics and natural language processing.|
|[WordNet Integration by TextBlob API](https://textblob.readthedocs.io/en/dev/quickstart.html#wordnet-integration)|You can access the synsets for a Word via the **synsets** property or the `get_synsets` method, optionally passing in a part of speech.|
|[PPDB](http://paraphrase.org/#/download)|PPDB is an automatically extracted database containing **millions paraphrases in 16 different languages**. The goal of PPBD is to improve language processing by making systems more robust to language variability and unseen words.|

## 🔹 Word-Embeddings Substitution

In this approach, we take pre-trained word embeddings such as Word2Vec, GloVe, FastText, Sent2Vec, and use the nearest neighbor words in the embedding space as the replacement for some word in the sentence.

[Jiao et al.](https://arxiv.org/abs/1909.10351) used this technique with GloVe embeddings in their paper _“TinyBert”_ to improve the generalization of their language model on downstream tasks. [Wang et al.](https://www.aclweb.org/anthology/D15-1306.pdf) used it to augment tweets needed to learn a topic model.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-embedding.png" width="365" height="278"/>

_For example,_ you can replace the word with the 3-most similar words and get three variations of the text.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-embedding-example.png" width="370" height="90"/>

It’s easy to use packages like **Gensim** to access pre-trained word vectors and get the nearest neighbors. _For example,_ here we find the synonyms for the word _**‘awesome’**_ using word vectors trained on tweets.

```python
# pip install gensim
import gensim.downloader as api

model = api.load('glove-twitter-25')  
model.most_similar('awesome', topn=5)
```

You will get back the 5 most similar words along with the cosine similarities.

```python
[('amazing', 0.9687871932983398),
 ('best', 0.9600659608840942),
 ('fun', 0.9331520795822144),
 ('fantastic', 0.9313924312591553),
 ('perfect', 0.9243415594100952)]
```

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)|Language model pre-training, such as BERT, has significantly improved the performances of many natural language processing tasks. However, pre-trained language models are usually computationally expensive, so it is difficult to efficiently execute them on resource-restricted devices. To accelerate inference and reduce model size while maintaining accuracy, we first propose a novel Transformer distillation method that is specially designed for knowledge distillation (KD) of the Transformer-based models. By leveraging this new KD method, the plenty of knowledge encoded in a large teacher BERT can be effectively transferred to a small student Tiny-BERT. Then, we introduce a new two-stage learning framework for TinyBERT, which performs Transformer distillation at both the pretraining and task-specific learning stages. This framework ensures that TinyBERT can capture he general-domain as well as the task-specific knowledge in BERT. TinyBERT with 4 layers is empirically effective and achieves more than 96.8% the performance of its teacher BERTBASE on GLUE benchmark, while being 7.5x smaller and 9.4x faster on inference. TinyBERT with 4 layers is also significantly better than 4-layer state-of-the-art baselines on BERT distillation, with only about 28% parameters and about 31% inference time of them. Moreover, TinyBERT with 6 layers performs on-par with its teacher BERTBASE.|
|[That’s So Annoying!!!: A Lexical and Frame-Semantic Embedding Based Data Augmentation Approach to Automatic Categorization of Annoying Behaviors using #petpeeve Tweets](https://aclanthology.org/D15-1306.pdf)|We propose a novel data augmentation ap- proach to enhance computational behav- ioral analysis using social media text. In particular, we collect a Twitter corpus of the descriptions of annoying behaviors us- ing the _#petpeeve_ hashtags. In the qual- itative analysis, we study the language use in these tweets, with a special focus on the fine-grained categories and the ge- ographic variation of the language. In quantitative analysis, we show that lexi- cal and syntactic features are useful for au- tomatic categorization of annoying behav- iors, and frame-semantic features further boost the performance; that leveraging large lexical embeddings to create addi- tional training instances significantly im- proves the lexical model; and incorporat- ing frame-semantic embedding achieves the best overall performance.|
|[Data Augmentation for Low-Resource Neural Machine Translation](https://arxiv.org/abs/1705.00440)|The quality of a Neural Machine Translation system depends substantially on the availability of sizable parallel corpora. For low-resource language pairs this is not the case, resulting in poor translation quality. Inspired by work in computer vision, we propose a novel data augmentation approach that targets low-frequency words by generating new sentence pairs containing rare words in new, synthetically created contexts. Experimental results on simulated low-resource settings show that our method improves translation quality by up to 2.9 BLEU points over the baseline and up to 3.2 BLEU over back-translation.|

## 🔹 Masked Language Model

Transformer models such as BERT, ROBERTA, and ALBERT have been trained on a large amount of text using a pretext task called “Masked Language Modeling” where the model has to predict masked words based on the context.

This can be used to augment some text. _For example,_ we could use a pre-trained BERT model, mask some parts of the text, and ask the BERT model to predict the token for the mask.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-bert-mlm.png" width="310" height="308"/>

Thus, we can generate variations of a text using the mask predictions. Compared to previous approaches, the generated text is more grammatically coherent as the model takes context into account when making predictions.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-bert-augmentations.png" width="549" height="165"/>

This is easy to implement with open-source libraries such as [transformers](https://huggingface.co/transformers/) by :hugs: Hugging Face. You can set the token you want to replace with `<mask>` and generate predictions.

```python
from transformers import pipeline

nlp = pipeline('fill-mask')
nlp('This is <mask> cool')
```
➡️ **Output:**

```python
[{'score': 0.515411913394928,
  'sequence': '<s> This is pretty cool</s>',
  'token': 1256},
 {'score': 0.1166248694062233,
  'sequence': '<s> This is really cool</s>',
  'token': 269},
 {'score': 0.07387523353099823,
  'sequence': '<s> This is super cool</s>',
  'token': 2422},
 {'score': 0.04272908344864845,
  'sequence': '<s> This is kinda cool</s>',
  'token': 24282},
 {'score': 0.034715913236141205,
  'sequence': '<s> This is very cool</s>',
  'token': 182}]
```

However, one caveat of this method is that deciding which part of the text to mask is not trivial. You will have to use heuristics to decide the mask, otherwise, the generated text might not retain the meaning of the original sentence.

[Garg. et al.](https://arxiv.org/abs/2004.01970) use this idea for generating adversarial examples for text classification.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/bae-adversarial-attack.png" width="667" height="351"/>

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[BAE: BERT-based Adversarial Examples for Text Classification](https://arxiv.org/abs/2004.01970)|Modern text classification models are susceptible to adversarial examples, perturbed versions of the original text indiscernible by humans which get misclassified by the model. Recent works in NLP use rule-based synonym replacement strategies to generate adversarial examples. These strategies can lead to out-of-context and unnaturally complex token replacements, which are easily identifiable by humans. We present BAE, a black box attack for generating adversarial examples using contextual perturbations from a BERT masked language model. BAE replaces and inserts tokens in the original text by masking a portion of the text and leveraging the BERT-MLM to generate alternatives for the masked tokens. Through automatic and human evaluations, we show that BAE performs a stronger attack, in addition to generating adversarial examples with improved grammaticality and semantic coherence as compared to prior work.|

## 🔹 TF-IDF based word replacement

This augmentation method was proposed by [Xie et al.](https://arxiv.org/abs/1904.12848) in the Unsupervised Data Augmentation paper. The basic idea is that words that have **low TF-IDF scores** are uninformative and thus can be replaced without affecting the ground-truth labels of the sentence.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-tf-idf-word-replacement.png" width="345" height="110"/>

The words that replace the original word are chosen by calculating TF-IDF scores of words over the whole document and taking the lowest ones. You can refer to the [code implementation](https://github.com/google-research/uda/blob/master/text/augmentation/word_level_augment.py) for this in the original paper.

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)|Semi-supervised learning lately has shown much promise in improving deep learning models when labeled data is scarce. Common among recent approaches is the use of consistency training on a large amount of unlabeled data to constrain model predictions to be invariant to input noise. In this work, we present a new perspective on how to effectively noise unlabeled examples and argue that the quality of noising, specifically those produced by advanced data augmentation methods, plays a crucial role in semi-supervised learning. By substituting simple noising operations with advanced data augmentation methods such as RandAugment and back-translation, our method brings substantial improvements across six language and three vision tasks under the same consistency training framework. On the IMDb text classification dataset, with only 20 labeled examples, our method achieves an error rate of 4.20, outperforming the state-of-the-art model trained on 25,000 labeled examples. On a standard semi-supervised learning benchmark, CIFAR-10, our method outperforms all previous approaches and achieves an error rate of 5.43 with only 250 examples. Our method also combines well with transfer learning, e.g., when finetuning from BERT, and yields improvements in high-data regime, such as ImageNet, whether when there is only 10% labeled data or when a full labeled set with 1.3M extra unlabeled examples is used. Code is available :octocat: [here](https://github.com/google-research/uda).|

# 💠 Back Translation

In this approach, we leverage machine translation to paraphrase a text while retraining the meaning. [Xie et al.](https://arxiv.org/abs/1904.12848) used this method to augment the unlabeled text and learn a semi-supervised model on IMDB dataset with only 20 labeled examples. Their model outperformed the previous state-of-the-art model trained on 25,000 labeled examples.

The back-translation process is as follows:

- Take some sentence _(e.g. in English)_ and translate to another Language _(e.g. French)_
- Translate the _French_ sentence back into an _English_ sentence
- Check if the new sentence is different from our original sentence. If it is, then we use this new sentence as an augmented version of the original text.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-back-translation.png" width="485" height="276"/>

You can also run back-translation using different languages at once to generate more variations. As shown below, we translate an English sentence to a target language and back again to English for three target languages: _French, Mandarin, and Italian_.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-backtranslation-multi.png" width="592" height="276"/>

This technique was also used in the [1st place solution](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52557) for the “Toxic Comment Classification Challenge” on Kaggle. The winner used it for both training-data augmentations as well as during test-time where the predicted probabilities for English sentence along with back-translation using three languages(French, German, Spanish) were averaged to get the final prediction.

For the implementation of back-translation, you can use TextBlob. Alternatively, you can also use [Google Sheets](https://amitness.com/2020/02/back-translation-in-google-sheets/) to apply Google Translate for free. You can also use MarianMT for back-translation.

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)|Semi-supervised learning lately has shown much promise in improving deep learning models when labeled data is scarce. Common among recent approaches is the use of consistency training on a large amount of unlabeled data to constrain model predictions to be invariant to input noise. In this work, we present a new perspective on how to effectively noise unlabeled examples and argue that the quality of noising, specifically those produced by advanced data augmentation methods, plays a crucial role in semi-supervised learning. By substituting simple noising operations with advanced data augmentation methods such as RandAugment and back-translation, our method brings substantial improvements across six language and three vision tasks under the same consistency training framework. On the IMDb text classification dataset, with only 20 labeled examples, our method achieves an error rate of 4.20, outperforming the state-of-the-art model trained on 25,000 labeled examples. On a standard semi-supervised learning benchmark, CIFAR-10, our method outperforms all previous approaches and achieves an error rate of 5.43 with only 250 examples. Our method also combines well with transfer learning, e.g., when finetuning from BERT, and yields improvements in high-data regime, such as ImageNet, whether when there is only 10% labeled data or when a full labeled set with 1.3M extra unlabeled examples is used. Code is available :octocat: [here](https://github.com/google-research/uda).|

### 📰 Articles

- [Back Translation for Text Augmentation with Google Sheets](https://amitness.com/2020/02/back-translation-in-google-sheets/)
- [Text Data Augmentation with MarianMT](https://amitness.com/back-translation/)

### 🛠️ Tools

| Title | Description, Information |
| :---:         |          :--- |
|[MarianNMT](https://marian-nmt.github.io)|Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies. It is mainly being developed by the Microsoft Translator team.|
|🤗 [HuggingFace - Traslation](https://huggingface.co/models?pipeline_tag=translation&sort=downloads)||

# 💠 Text Surface Transformation

These are simple pattern matching transformations applied using regex and was introduced by Claude Coulombe in his paper - ["Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs"](https://arxiv.org/abs/1812.04718).

In the paper, he gives an example of transforming verbal forms from contraction to expansion and vice versa. We can generate augmented texts by applying this.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-contraction.png" width="590" height="85"/>

Since the transformation should not change the meaning of the sentence, we can see that this can fail in case of expanding ambiguous verbal forms like:

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-contraction-ambiguity.png" width="619" height="209"/>

To resolve this, the paper proposes that we allow ambiguous contractions but skip ambiguous expansion.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-contraction-solution.png" width="513" height="262"/>

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs](https://arxiv.org/abs/1812.04718) by Claude Coulombe|In practice, it is common to find oneself with far too little text data to train a deep neural network. This "Big Data Wall" represents a challenge for minority language communities on the Internet, organizations, laboratories and companies that compete the GAFAM (Google, Amazon, Facebook, Apple, Microsoft). While most of the research effort in text data augmentation aims on the long-term goal of finding end-to-end learning solutions, which is equivalent to "using neural networks to feed neural networks", this engineering work focuses on the use of practical, robust, scalable and easy-to-implement data augmentation pre-processing techniques similar to those that are successful in computer vision. Several text augmentation techniques have been experimented. Some existing ones have been tested for comparison purposes such as noise injection or the use of regular expressions. Others are modified or improved techniques like lexical replacement. Finally more innovative ones, such as the generation of paraphrases using back-translation or by the transformation of syntactic trees, are based on robust, scalable, and easy-to-use NLP Cloud APIs. All the text augmentation techniques studied, with an amplification factor of only 5, increased the accuracy of the results in a range of 4.3% to 21.6%, with significant statistical fluctuations, on a standardized task of text polarity prediction. Some standard deep neural network architectures were tested: the multilayer perceptron (MLP), the long short-term memory recurrent network (LSTM) and the bidirectional LSTM (biLSTM). Classical XGBoost algorithm has been tested with up to 2.5% improvements.|

### 🛠️ Tools

| Title | Description, Information |
| :---:         |          :--- |
|[Wikipedia:List of English contractions](https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions)|This is a list of contractions used in the Wikipedia:Manual of Style/Abbreviations; these are to be avoided anywhere other than in direct quotations in encyclopedic prose.|
|:octocat: [contractions](https://github.com/kootenpv/contractions)|This package is capable of resolving contractions and slang.|

# 💠 Random Noise Injection

The idea of these methods is to inject noise in the text so that the model trained is robust to perturbations.

## 🔹 Spelling Error Injection

In this method, we add spelling errors to some random word in the sentence. These spelling errors can be added programmatically or using a mapping of common spelling errors.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-spelling-example.png" width="475" height="87"/>

## 🔹 QWERTY Keyboard Error Injection

This method tries to simulate common errors that happen when typing on a QWERTY layout keyboard due to keys that are very near to each other. The errors are injected based on keyboard distance.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-keyboard-error-example.png" width="485" height="225"/>

## 🔹 Unigram Noising

This method has been used by [Xie et al.](https://arxiv.org/abs/1703.02573) and the [Unsupervised Data Augmentation for Consistency Training (UDA)](https://arxiv.org/abs/1904.12848) paper. 

The idea is to perform replacement with words sampled from the unigram frequency distribution. This frequency is basically how many times each word occurs in the training corpus.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-unigram-noise.png" width="655" height="294"/>

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Data Noising as Smoothing in Neural Network Language Models](https://arxiv.org/abs/1703.02573)|Data noising is an effective technique for regularizing neural network models. While noising is widely adopted in application domains such as vision and speech, commonly used noising primitives have not been developed for discrete sequence-level settings such as language modeling. In this paper, we derive a connection between input noising in neural network language models and smoothing in n-gram models. Using this connection, we draw upon ideas from smoothing to develop effective noising schemes. We demonstrate performance gains when applying the proposed schemes to language modeling and machine translation. Finally, we provide empirical analysis validating the relationship between noising and smoothing.|
|[Unsupervised Data Augmentation for Consistency Training (UDA)](https://arxiv.org/abs/1904.12848)|Semi-supervised learning lately has shown much promise in improving deep learning models when labeled data is scarce. Common among recent approaches is the use of consistency training on a large amount of unlabeled data to constrain model predictions to be invariant to input noise. In this work, we present a new perspective on how to effectively noise unlabeled examples and argue that the quality of noising, specifically those produced by advanced data augmentation methods, plays a crucial role in semi-supervised learning. By substituting simple noising operations with advanced data augmentation methods such as RandAugment and back-translation, our method brings substantial improvements across six language and three vision tasks under the same consistency training framework. On the IMDb text classification dataset, with only 20 labeled examples, our method achieves an error rate of 4.20, outperforming the state-of-the-art model trained on 25,000 labeled examples. On a standard semi-supervised learning benchmark, CIFAR-10, our method outperforms all previous approaches and achieves an error rate of 5.43 with only 250 examples. Our method also combines well with transfer learning, e.g., when finetuning from BERT, and yields improvements in high-data regime, such as ImageNet, whether when there is only 10% labeled data or when a full labeled set with 1.3M extra unlabeled examples is used. Code is available :octocat: [here](https://github.com/google-research/uda).|


## 🔹 Blank Noising

This method has been proposed by [Xie et al.](https://arxiv.org/abs/1703.02573) in their paper - "Data Noising as Smoothing in Neural Network Language Models". The idea is to replace a random word with a placeholder token. The paper uses “_” as the placeholder token. In the paper, they use it as a way to avoid overfitting on specific contexts as well as a smoothing mechanism for the language model. The technique helped improve perplexity and BLEU scores.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-blank-noising.png" width="635" height="42"/>

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Data Noising as Smoothing in Neural Network Language Models](https://arxiv.org/abs/1703.02573)|Data noising is an effective technique for regularizing neural network models. While noising is widely adopted in application domains such as vision and speech, commonly used noising primitives have not been developed for discrete sequence-level settings such as language modeling. In this paper, we derive a connection between input noising in neural network language models and smoothing in n-gram models. Using this connection, we draw upon ideas from smoothing to develop effective noising schemes. We demonstrate performance gains when applying the proposed schemes to language modeling and machine translation. Finally, we provide empirical analysis validating the relationship between noising and smoothing.|

## 🔹 Sentence Shuffling

This is a naive technique where we shuffle sentences present in a training text to create an augmented version.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-sentence-shuffle.png" width="501" height="201"/>

## 🔹 Random Insertion

This technique was proposed by Wei et al. in their paper [“Easy Data Augmentation”](https://arxiv.org/abs/1901.11196). 

In this technique, we first choose a random word from the sentence that is not a stop word. Then, we find its synonym and insert that into a random position in the sentence.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-random-insertion.png" width="675" height="201"/>

_For example,_ given the sentence:

_This **article** will focus on summarizing data augmentation **techniques** in NLP._

The method randomly selects n words (say two), the words **_article_** and **_techniques_** find the synonyms as **_write-up_** and **_methods_** respectively. Then these synonyms are inserted at a random position in the sentence.

_This article will focus on **write-up** summarizing data augmentation techniques in NLP **methods**._

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)|We present EDA: easy data augmentation techniques for boosting performance on text classification tasks. EDA consists of four simple but powerful operations: synonym replacement, random insertion, random swap, and random deletion. On five text classification tasks, we show that EDA improves performance for both convolutional and recurrent neural networks. EDA demonstrates particularly strong results for smaller datasets; on average, across five datasets, training with EDA while using only 50% of the available training set achieved the same accuracy as normal training with all available data. We also performed extensive ablation studies and suggest parameters for practical use.|

## 🔹 Random Swap

This technique was also proposed by Wei et al. in their paper [“Easy Data Augmentation”](https://arxiv.org/abs/1901.11196). 

The idea is to randomly swap any two words in the sentence.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-random-swap.png" width="538" height="89"/>

## 🔹 Random Deletion

This technique was also proposed by Wei et al. in their paper [“Easy Data Augmentation”](https://arxiv.org/abs/1901.11196). 

In this, we randomly remove each word in the sentence with some probability _p_.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-random-deletion.png" width="523" height="88"/>

# 💠 Instance Crossover Augmentation

This technique was introduced by [Luque](https://arxiv.org/abs/1909.11241) in his paper on sentiment analysis for TASS 2019. It is inspired by the chromosome crossover operation that happens in genetics. 

In the method, a tweet is divided into two halves and two random tweets of the same polarity _(i.e. positive/negative)_ have their halves swapped. The hypothesis is that even though the result will be ungrammatical and semantically unsound, the new text will still preserve the sentiment.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-instance-crossover.png" width="717" height="291"/>

This technique had no impact on the accuracy but helped with the F1 score in the paper showing that it helps minority classes such as the Neutral class with fewer tweets.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-instance-crossover-result.png" width="677" height="126"/>

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Atalaya at TASS 2019: Data Augmentation and Robust Embeddings for Sentiment Analysis](https://arxiv.org/abs/1909.11241)|In this article we describe our participation in TASS 2019, a shared task aimed at the detection of sentiment polarity of Spanish tweets. We combined different representations such as bag-of-words, bag-of-characters, and tweet embeddings. In particular, we trained robust subword-aware word embeddings and computed tweet representations using a weighted-averaging strategy. We also used two data augmentation techniques to deal with data scarcity: two-way translation augmentation, and instance crossover augmentation, a novel technique that generates new instances by combining halves of tweets. In experiments, we trained linear classifiers and ensemble models, obtaining highly competitive results despite the simplicity of our approaches.|

# 💠 Albumentation

In this section, we will see how we can apply some of the ideas used in CV data augmentation in NLP. 

For that, we will use the [Albumentations](https://github.com/albumentations-team/albumentations) package. 

## 🔹 Shuffle Sentences Transform

In this transformation, if the given text sample contains multiple sentences these sentences are shuffled to create a new sample. 

_For example:_

`text = ‘<Sentence1>. <Sentence2>. <Sentence4>. <Sentence4>. <Sentence5>. <Sentence5>.’`

Is transformed to:

`text = ‘<Sentence2>. <Sentence3>. <Sentence1>. <Sentence5>. <Sentence5>. <Sentence4>.’`

## 🔹 Exclude duplicate transform

In this transformation, if the given text sample contains multiple sentences with duplicate sentences, these duplicate sentences are removed to create a new sample.

_For example_ given the sample,

`text = ‘<Sentence1>. <Sentence2>. <Sentence4>. <Sentence4>. <Sentence5>. <Sentence5>.’`

We transform it to:

`text = ‘<Sentence1>. <Sentence2>.<Sentence4>. <Sentence5>.’`

There are many other transformations which you can try with this library. You can check this wonderful [notebook](https://www.kaggle.com/code/shonenkov/nlp-albumentations/notebook) to see the complete implementation.

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125)|<p>Data augmentation is a commonly used technique for increasing both the size and the diversity of labeled training sets by leveraging input transformations that preserve corresponding output labels. In computer vision, image augmentations have become a common implicit regularization technique to combat overfitting in deep learning models and are ubiquitously used to improve performance. While most deep learning frameworks implement basic image transformations, the list is typically limited to some variations of flipping, rotating, scaling, and cropping. Moreover, image processing speed varies in existing image augmentation libraries. We present Albumentations, a fast and flexible open source library for image augmentation with many various image transform operations available that is also an easy-to-use wrapper around other augmentation libraries. We discuss the design principles that drove the implementation of Albumentations and give an overview of the key features and distinct capabilities. Finally, we provide examples of image augmentations for different computer vision tasks and demonstrate that Albumentations is faster than other commonly used image augmentation tools on most image transform operations.</p><ul><li>[Albumentations](https://albumentations.ai) - Albumentations is a computer vision tool that boosts the performance of deep convolutional neural networks. The library is widely used in industry, deep learning research, machine learning competitions, and open source projects.</li><li>[Albumentations documentation](https://albumentations.ai/docs/)</li><li> :octocat: [Albumentations](https://github.com/albumentations-team/albumentations) - Albumentations is a Python library for image augmentation. Image augmentation is used in deep learning and computer vision tasks to increase the quality of trained models. The purpose of image augmentation is to create new training samples from the existing data.</li><li>[NLP Albumentations](https://www.kaggle.com/code/shonenkov/nlp-albumentations/notebook) notebook on Kagle</li></ul>|

# 💠 Syntax-tree Manipulation

This technique has been used in the paper by [Coulombe](https://arxiv.org/abs/1812.04718). The idea is to parse and generate the dependency tree of the original sentence, transform it using rules, and generate a paraphrased sentence.

_For example,_ one transformation that doesn’t change the meaning of the sentence is the transformation from active voice to the passive voice of sentence and vice versa.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-syntax-tree-manipulation.png" width="821" height="426"/>

### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs](https://arxiv.org/abs/1812.04718)|In practice, it is common to find oneself with far too little text data to train a deep neural network. This "Big Data Wall" represents a challenge for minority language communities on the Internet, organizations, laboratories and companies that compete the GAFAM (Google, Amazon, Facebook, Apple, Microsoft). While most of the research effort in text data augmentation aims on the long-term goal of finding end-to-end learning solutions, which is equivalent to "using neural networks to feed neural networks", this engineering work focuses on the use of practical, robust, scalable and easy-to-implement data augmentation pre-processing techniques similar to those that are successful in computer vision. Several text augmentation techniques have been experimented. Some existing ones have been tested for comparison purposes such as noise injection or the use of regular expressions. Others are modified or improved techniques like lexical replacement. Finally more innovative ones, such as the generation of paraphrases using back-translation or by the transformation of syntactic trees, are based on robust, scalable, and easy-to-use NLP Cloud APIs. All the text augmentation techniques studied, with an amplification factor of only 5, increased the accuracy of the results in a range of 4.3% to 21.6%, with significant statistical fluctuations, on a standardized task of text polarity prediction. Some standard deep neural network architectures were tested: the multilayer perceptron (MLP), the long short-term memory recurrent network (LSTM) and the bidirectional LSTM (biLSTM). Classical XGBoost algorithm has been tested with up to 2.5% improvements.|

# 💠 MixUp for Text

Mixup is a simple yet effective image augmentation technique introduced by [Zhang et al.](https://arxiv.org/abs/1710.09412) in 2017. The idea is to combine two random images in a mini-batch in some proportion to generate synthetic examples for training. For images, this means combining image pixels of two different classes. It acts as a form of regularization during training.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-mixup-image.png" width="513" height="170"/>

Bringing this idea to NLP, [Guo et al.](https://arxiv.org/abs/1905.08941) modified Mixup to work with text. They propose two novel approaches for applying Mixup to text:

## 🔹 wordMixup

In this method, two random sentences in a mini-batch are taken and they are zero-padded to the same length. Then, their word embeddings are combined in some proportion. The resulting word embedding is passed to the usual flow for text classification. The cross-entropy loss is calculated for both the labels of the original text in the given proportion.

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/Data%20Augmentation/img/nlp-aug-wordmixup.png" width="806" height="199"/>

## 🔹 sentMixup

In this method, two sentences are taken and they are zero-padded to the same length. Then, their word embeddings are passed through LSTM/CNN encoder and we take the last hidden state as sentence embedding. These embeddings are combined in a certain proportion and then passed to the final classification layer. The cross-entropy loss is calculated based on both the labels of original sentences in the given proportion.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-sentmixup.png" width="836" height="219"/>

## 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Augmenting Data with Mixup for Sentence Classification: An Empirical Study](https://arxiv.org/abs/1905.08941)|Mixup, a recent proposed data augmentation method through linearly interpolating inputs and modeling targets of random samples, has demonstrated its capability of significantly improving the predictive accuracy of the state-of-the-art networks for image classification. However, how this technique can be applied to and what is its effectiveness on natural language processing (NLP) tasks have not been investigated. In this paper, we propose two strategies for the adaption of Mixup on sentence classification: one performs interpolation on word embeddings and another on sentence embeddings. We conduct experiments to evaluate our methods using several benchmark datasets. Our studies show that such interpolation strategies serve as an effective, domain independent data augmentation approach for sentence classification, and can result in significant accuracy improvement for both CNN and LSTM models.|
|[Mixup-Transfomer: Dynamic Data Augmentation for NLP Tasks](https://www.semanticscholar.org/reader/2a8a2ab581f2e89c9a66e1b353346e1bb86ee6f6)|Mixup (Zhang et al., 2017) is a latest data augmentation technique that linearly interpolates input examples and the corresponding labels. It has shown strong effectiveness in image classification byinterpolating images at the pixel level. Inspired by this line of research, in this paper, we explore: i) how to apply mixup to natural language processing tasks since text data can hardly be mixedin the raw format; ii) if mixup is still effective in transformer-based learning models, e.g., BERT. To achieve the goal, we incorporate mixup to transformer-based pre-trained architecture, named“mixup-transformer”, for a wide range of NLP tasks while keeping the whole end-to-end training system. We evaluate the proposed framework by running extensive experiments on the GLUE benchmark. Furthermore, we also examine the performance of mixup-transformer in low-resource scenarios by reducing the training data with a certain ratio. Our studies show that mixup is a domain-independent data augmentation technique to pre-trained language models, resulting in significant performance improvement for transformer-based models.|

# 💠 Generative Methods

This line of work tries to generate additional training data while preserving the class label.

## 🔹 Conditional Pre-trained Language Models

This technique was first proposed by Anaby-Tavor et al. in their paper [“Not Enough Data? Deep Learning to the Rescue!](https://arxiv.org/abs/1911.03118). A recent paper from [Kumar et al.](https://arxiv.org/abs/2003.02245) evaluated this idea across multiple transformer-based pre-trained models. The problem formulation is as follows:

- Prepend the class label to each text in your training data

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-generation-training.png" width="736" height="187"/>

- Fine-tune a large pre-trained language model (BERT, GPT-3, BART) on this modified training data. For GPT-3, the fine-tuning task is generation while for BERT, the goal would be a masked token prediction.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-gpt-3-finetuning.png" width="453" height="213.3"/>

- Using the fine-tuned language model, new samples can be generated by using the class label and a few initial words as the prompt for the model. The paper uses 3 initial words of each training text and also generates one synthetic example for each point in the training data.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Data%20Augmentation/img/nlp-aug-gpt-3.png" width="457.3" height="195.3"/>

### 📰 Articles

- [Text Data Augmentation Using the GPT-2 Language Model](https://towardsdatascience.com/text-data-augmentation-using-gpt-2-language-model-e5384e15b550)


### 📄 Papers

| Title | Description, Information |
| :---:         |          :--- |
|[Not Enough Data? Deep Learning to the Rescue!](https://arxiv.org/abs/1911.03118)|Based on recent advances in natural language modeling and those in text generation capabilities, we propose a novel data augmentation method for text classification tasks. We use a powerful pre-trained neural network model to artificially synthesize new labeled data for supervised learning. We mainly focus on cases with scarce labeled data. Our method, referred to as language-model-based data augmentation (LAMBADA), involves fine-tuning a state-of-the-art language generator to a specific task through an initial training phase on the existing (usually small) labeled data. Using the fine-tuned model and given a class label, new sentences for the class are generated. Our process then filters these new sentences by using a classifier trained on the original data. In a series of experiments, we show that LAMBADA improves classifiers' performance on a variety of datasets. Moreover, LAMBADA significantly improves upon the state-of-the-art techniques for data augmentation, specifically those applicable to text classification tasks with little data.|
|[Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/abs/2003.02245)|Language model based pre-trained models such as BERT have provided significant gains across different NLP tasks. In this paper, we study different types of transformer based pre-trained models such as auto-regressive models (GPT-2), auto-encoder models (BERT), and seq2seq models (BART) for conditional data augmentation. We show that prepending the class labels to text sequences provides a simple yet effective way to condition the pre-trained models for data augmentation. Additionally, on three classification benchmarks, pre-trained Seq2Seq model outperforms other data augmentation methods in a low-resource setting. Further, we explore how different pre-trained model based data augmentation differs in-terms of data diversity, and how well such methods preserve the class-label information.|

###  ⚙️ Tools

| Title | Description, Information |
| :---:         |          :--- |
|[TextAugmentation-GPT2](https://github.com/prakhar21/TextAugmentation-GPT2)|Fine-tuned pre-trained GPT2 for custom topic specific text generation. Such system can be used for Text Augmentation.|

# 💠 Paraphraser

## 📰 Articles

- [NLP Data Augmentation](https://pemagrg.medium.com/nlp-data-augmentation-a346479b295f)

# ‼️ Things to keep in mind while doing NLP data augmentation

The **main issue faced when training on augmented data** is that algorithms, when done incorrectly, is that you heavily overfit the augmented training data.

**_Some things to keep in mind:_**

- Do not validate using the augmented data.
- If you’re doing K-fold cross-validation, always keep the original sample and augmented sample in the same fold to avoid overfitting.
- Always try different augmentation approaches and check which works better.
- A mix of different augmentation methods is also appreciated but don’t overdo it.
- Experiment to determine the optimal number of samples to be augmented to get the best results.
- Keep in mind that data augmentation in NLP does not always help to improve model performance.

# ❓How much augmentation?

Finally, how many augmented sentences should we generate for the real sentence? The answer for this depends on the size of your dataset. If you only have a small dataset, overfitting is more likely so you can generate a larger number of augmented sentences. For larger datasets, adding too much augmented data can be unhelpful since your model may already be able to generalize when there is a large amount of real data. 

# 📄 Papers

- [Papers with Code - Data Augmentation](https://paperswithcode.com/task/data-augmentation)
- [Papers with Code - Text Augmentation](https://paperswithcode.com/task/text-augmentation)
- :octocat: [Data Augmentation Techniques for NLP](https://github.com/styfeng/DataAug4NLP)

| Title | Description, Information |
| :---:         |          :--- |
|[Curriculum Data Augmentation for Highly Multiclass Text Classification](https://www.microsoft.com/en-us/research/publication/curriculum-data-augmentation-for-highly-multiclass-text-classification/), 19th Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2021), March 2021|This paper explores data augmentation—a technique particularly suitable for training with limited data—for highly multiclass text classification tasks, which have a large number of output classes. On four diverse highly multi-class tasks, we find that well-known data augmentation techniques (Sennrich et al., 2016b;Wang et al., 2018; Wei and Zou, 2019) can improve performance by up to 3.0% on average. To further boost performance, we present a simple training strategy called curriculum data augmentation, which leverages curriculum learning by first training on only original examples and then introducing augmented data as training progresses. We explore a two-stage and a gradual schedule, and find that, compared with standard single-stage training, curriculum data augmentation improves performance, trains faster, and maintains robustness high augmentation temperatures (strengths)|
|[A Survey of Data Augmentation Approaches for NLP](https://arxiv.org/abs/2105.03075)|Data augmentation has recently seen increased interest in NLP due to more work in low-resource domains, new tasks, and the popularity of large-scale neural networks that require large amounts of training data. Despite this recent upsurge, this area is still relatively underexplored, perhaps due to the challenges posed by the discrete nature of language data. In this paper, we present a comprehensive and unifying survey of data augmentation for NLP by summarizing the literature in a structured manner. We first introduce and motivate data augmentation for NLP, and then discuss major methodologically representative approaches. Next, we highlight techniques that are used for popular NLP applications and tasks. We conclude by outlining current challenges and directions for future research. Overall, our paper aims to clarify the landscape of existing literature in data augmentation for NLP and motivate additional work in this area.|
|[Text Data Augmentation for Deep Learning](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00492-0)|Natural Language Processing (NLP) is one of the most captivating applications of Deep Learning. In this survey, we consider how the Data Augmentation training strategy can aid in its development. We begin with the major motifs of Data Augmentation summarized into strengthening local decision boundaries, brute force training, causality and counterfactual examples, and the distinction between meaning and form. We follow these motifs with a concrete list of augmentation frameworks that have been developed for text data. Deep Learning generally struggles with the measurement of generalization and characterization of overfitting. We highlight studies that cover how augmentations can construct test sets for generalization. NLP is at an early stage in applying Data Augmentation compared to Computer Vision. We highlight the key differences and promising ideas that have yet to be tested in NLP. For the sake of practical implementation, we describe tools that facilitate Data Augmentation such as the use of consistency regularization, controllers, and offline and online augmentation pipelines, to preview a few. Finally, we discuss interesting topics around Data Augmentation in NLP such as task-specific augmentations, the use of prior knowledge in self-supervised learning versus Data Augmentation, intersections with transfer and multi-task learning, and ideas for AI-GAs (AI-Generating Algorithms). We hope this paper inspires further research interest in Text Data Augmentation.|
|[Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://arxiv.org/abs/1805.06201)|<p>We propose a novel data augmentation for labeled sentences called contextual augmentation. We assume an invariance that sentences are natural even if the words in the sentences are replaced with other words with paradigmatic relations. We stochastically replace words with other words that are predicted by a bi-directional language model at the word positions. Words predicted according to a context are numerous but appropriate for the augmentation of the original words. Furthermore, we retrofit a language model with a label-conditional architecture, which allows the model to augment sentences without breaking the label-compatibility. Through the experiments for six various different text classification tasks, we demonstrate that the proposed method improves classifiers based on the convolutional or recurrent neural networks.</p><ul><li> :octocat: [Contextual Augmentation](https://github.com/pfnet-research/contextual_augmentation)</li></ul>|
|[TABAS: Text augmentation based on attention score for text classification model](https://www.sciencedirect.com/science/article/pii/S2405959521001454)|To improve the performance of text classification, we propose text augmentation based on attention score (TABAS). We recognized that a criterion for selecting a replacement word rather than a random selection was necessary. Therefore, TABAS utilizes attention scores for text modification, processing only words with the same entity and part-of-speech tags to consider informational aspects. To verify this approach, we used two benchmark tasks. As a result, TABAS can significantly improve performance, both recurrent and convolutional neural networks. Furthermore, we confirm that it provides a practical way to develop deep-learning models by saving costs on making additional datasets.|
|[Data augmentation approaches in natural language processing: A survey](https://www.sciencedirect.com/science/article/pii/S2666651022000080)|As an effective strategy, data augmentation (DA) alleviates data scarcity scenarios where deep learning techniques may fail. It is widely applied in computer vision then introduced to natural language processing and achieves improvements in many tasks. One of the main focuses of the DA methods is to improve the diversity of training data, thereby helping the model to better generalize to unseen testing data. In this survey, we frame DA methods into three categories based on the diversity of augmented data, including paraphrasing, noising, and sampling. Our paper sets out to analyze DA methods in detail according to the above categories. Further, we also introduce their applications in NLP tasks as well as the challenges. Some useful resources are provided in Appendix A.|

# 📰 Articles

- [Data Augmentation](https://en.wikipedia.org/wiki/Data_augmentation) on Wikipedia
- [A Visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)
- [Data Augmentation in NLP: Best Practices From a Kaggle Master](https://neptune.ai/blog/data-augmentation-nlp) - Collection of papers and resources for data augmentation for NLP.
- [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28)
- [Data Augmentation library for text](https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
- [How to Perform Data Augmentation in NLP Projects](https://www.freecodecamp.org/news/how-to-perform-data-augmentation-in-nlp-projects/)
- [Text Data Augmentation in Natural Language Processing with Texattack](https://www.analyticsvidhya.com/blog/2022/02/text-data-augmentation-in-natural-language-processing-with-texattack/)
- [Popular Data Augmentation Techniques in NLP](https://blog.paperspace.com/data-augmentation-for-nlp/)
- [How does Data Noising Help to Improve your NLP Model?](https://pub.towardsai.net/how-does-data-noising-help-to-improve-your-nlp-model-480619f9fb10)
- [NLP Data Augmentation](https://pemagrg.medium.com/nlp-data-augmentation-a346479b295f)

# 🛠️ Libraries, frameworks, etc.

| Title | Description, Information |
| :---:         |          :--- |
|[nlpaug](https://github.com/makcedward/nlpaug)|This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28). `Augmenter` is the basic element of augmentation while `Flow` is a pipeline to orchestra multi augmenter together.|
|[TextAugment](https://github.com/dsfsi/textaugment)|TextAugment is a Python 3 library for augmenting text for natural language processing applications. TextAugment stands on the giant shoulders of NLTK, Gensim, and TextBlob and plays nicely with them.|
|[TextAttack 🐙](https://github.com/QData/TextAttack)|TextAttack is a Python framework for adversarial attacks, data augmentation, and model training in NLP.|
|[TextFlint](https://github.com/textflint/textflint#usage)|<p>**Unified Multilingual Robustness Evaluation Toolkit for Natural Language Processing**</p><p>TextFlint is a multilingual robustness evaluation platform for natural language processing, which unifies text transformation, sub-population, adversarial attack,and their combinations to provide a comprehensive robustness analysis. So far, TextFlint supports 13 NLP tasks.</p><ul><li> 📄 **Paper:** [TextFlint: Unified Multilingual Robustness Evaluation Toolkit for Natural Language Processing](https://aclanthology.org/2021.acl-demo.41.pdf)</li><li>[TextFlint Documentation](https://textflint.readthedocs.io/en/latest/)</li></ul>|
|[AugLy](https://github.com/facebookresearch/AugLy)|<p>AugLy is a data augmentations library that currently supports four modalities (audio, image, text & video) and over 100 augmentations. Each modality’s augmentations are contained within its own sub-library. These sub-libraries include both function-based and class-based transforms, composition operators, and have the option to provide metadata about the transform applied, including its intensity.</p><ul><li> 📄 **Paper:** [AugLy: Data Augmentations for Robustness](https://arxiv.org/abs/2201.06494)</li><li>[AugLy’s documentation](https://augly.readthedocs.io/en/latest/)</li><li>[AugLy: A new data augmentation library to help build more robust AI models](https://ai.facebook.com/blog/augly-a-new-data-augmentation-library-to-help-build-more-robust-ai-models/)</li></ul>|
|[NL-Augmenter 🦎 → 🐍](https://github.com/GEM-benchmark/NL-Augmenter)|The NL-Augmenter is a collaborative effort intended to add transformations of datasets dealing with natural language. Transformations augment text datasets in diverse ways, including: randomizing names and numbers, changing style/syntax, paraphrasing, KB-based paraphrasing ... and whatever creative augmentation you contribute.|
|[Snorkel](https://github.com/snorkel-team/snorkel)|<ul><li>[📈 Snorkel Intro Tutorial: Data Augmentation](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/02_spam_data_augmentation_tutorial.ipynb)</li><li> :octocat: [Snorkel Tutorials](https://github.com/snorkel-team/snorkel-tutorials)</li><li>[Tutorials](https://www.snorkel.org/use-cases/)</li></ul>|
|[NLTK :: Sample usage for wordnet](https://www.nltk.org/howto/wordnet.html)|NLTK provides a programmatic access to WordNet|
|[MarianNMT](https://marian-nmt.github.io)|Marian is an efficient, free Neural Machine Translation framework written in pure C++ with minimal dependencies. It is mainly being developed by the Microsoft Translator team.|
|[Parrot Paraphraser](https://github.com/PrithivirajDamodaran/Parrot_Paraphraser)|A practical and feature-rich paraphrasing framework to augment human intents in text form to build robust NLU models for conversational engines.|
|[Pegasus Paraphraser](https://huggingface.co/tuner007/pegasus_paraphrase)|PEGASUS is a standard Transformer encoder-decoder. PEGASUS uses GSG to pre-train a Transformer encoder-decoder on large corpora of documents.|
|🤗 [HuggingFace - Traslation](https://huggingface.co/models?pipeline_tag=translation&sort=downloads)||
|:octocat: [contractions](https://github.com/kootenpv/contractions)|This package is capable of resolving contractions and slang.|


# ⚙️ Tools

| Title | Description, Information |
| :---:         |          :--- |
|[WordNet](https://wordnet.princeton.edu)|WordNet is a large lexical database of English. Nouns, verbs, adjectives and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations. The resulting network of meaningfully related words and concepts can be navigated with the [browser](http://wordnetweb.princeton.edu/perl/webwn). WordNet is also freely and publicly available for [download](https://wordnet.princeton.edu/node/5). WordNet's structure makes it a useful tool for computational linguistics and natural language processing.|
|[WordNet Integration by TextBlob API](https://textblob.readthedocs.io/en/dev/quickstart.html#wordnet-integration)|You can access the synsets for a Word via the **synsets** property or the `get_synsets` method, optionally passing in a part of speech.|
|[PPDB](http://paraphrase.org/#/download)|PPDB is an automatically extracted database containing **millions paraphrases in 16 different languages**. The goal of PPBD is to improve language processing by making systems more robust to language variability and unseen words.|
|[Wikipedia:List of English contractions](https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions)|This is a list of contractions used in the Wikipedia:Manual of Style/Abbreviations; these are to be avoided anywhere other than in direct quotations in encyclopedic prose.|
|[Natural Language Processing](https://github.com/AgaMiko/data-augmentation-review/blob/master/README.md#natural-language-processing)|List of useful data augmentation resources. You will find here some links to more or less popular github repos ✨, libraries, papers 📚 and other information.|


- [data-aug-experiment.ipynb](https://colab.research.google.com/drive/1eZlrxmXoNpjhd0CEoZjVnEkHHuy66ZM8?usp=sharing#scrollTo=27pnGpPgKL2u), Google Colaboratory Notebook

# :octocat: GitHub Repositories

| Title | Description, Information |
| :---:         |          :--- |
|[Data Augmentation Techniques for NLP](https://github.com/styfeng/DataAug4NLP)|<p>Collection of papers and resources for data augmentation for NLP.</p><p>Papers grouped by _text classification, translation, summarization, question-answering, sequence tagging, parsing, grammatical-error-correction, generation, dialogue, multimodal, mitigating bias, mitigating class imbalance, adversarial examples, compositionality, and automated augmentation._</p>|
|[Natural Language Processing](https://github.com/AgaMiko/data-augmentation-review/blob/master/README.md#natural-language-processing)|List of useful data augmentation resources. You will find here some links to more or less popular github repos ✨, libraries, papers 📚 and other information.|
