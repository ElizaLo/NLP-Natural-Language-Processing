# Fine Tune BERT for Text Classification with TensorFlow

> [Coursera Project Network](https://www.coursera.org/projects/fine-tune-bert-tensorflow?utm_source=gg&utm_medium=sem&utm_content=01-CatalogDSA-ML2-US&campaignid=12490862811&adgroupid=119269357576&device=c&keyword=&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=503940597764&hide_mobile_promo&gclid=Cj0KCQjw1PSDBhDbARIsAPeTqre_pVyk8xdwjiyXYUfSXAJ_NzCbPYD_Sv0Zio3iirRJRfcAzglJbpQaAmMrEALw_wcB)

## Course Objectives

In this course, we are going to focus on three learning objectives:

- Build TensorFlow Input Pipelines for Text Data with the [tf.data](https://www.tensorflow.org/api_docs/python/tf/data) API 
- Tokenize and Preprocess Text for [BERT](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2)
- Fine-tune BERT for text classification with TensorFlow and [TensorFlow Hub](https://tfhub.dev/)

This is a guided project on fine-tuning a Bidirectional Transformers for Language Understanding (BERT) model for text classification with TensorFlow. In this 2 hour long project, you will learn to preprocess and tokenize data for BERT classification, build TensorFlow input pipelines for text data with the tf.data API, and train and evaluate a fine-tuned BERT model for text classification with TensorFlow 2 and TensorFlow Hub. 

## Resources on how BERT works
In the hands-on component of this project, we will be focused on applying a BERT model from TF Hub for text classification. As such, we will not be covering the details of how BERT works. To complement the application based skills you will gain in the hands-on component, here are a few terrific resources on Transformer architectures and BERT. Please do make sure to spend some time reading them before continuing:

1. [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

2. [Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)

3. [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/)

4. [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

5. For more advanced learners, here's the original BERT paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

6. (Optional) [Sentiment Analysis with Deep Learning using BERT](https://www.coursera.org/projects/sentiment-analysis-bert) (PyTorch)



Inputs to BERT have to be tokenized before inference. In what order are the following steps supposed to be performed?

1. Split input string into tokens
2. Append [CLS] and prepend [SEP] tokens
3. Substitute tokens with their ids
