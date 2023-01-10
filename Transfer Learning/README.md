# Transfer Learning


Transfer learning is a methodology where weights from a model trained on one task are taken and either used to construct a fixed feature extractor, as weight initialization and/or fine-tuning.

## 📄 Papers

- [Papers with Code - Transfer Learning](https://paperswithcode.com/task/transfer-learning)

| Title | Description, Information |
| :---:         |          :--- |
|**Universal Language Model Fine-tuning for Text Classification (ULMFiT)**|<p> 📄 **Paper:** [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) by Jeremy Howard, Sebastian Ruder</p><p>Inductive transfer learning has greatly impacted computer vision, but existing approaches in NLP still require task-specific modifications and training from scratch. We propose Universal Language Model Fine-tuning (ULMFiT), an effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model. Our method significantly outperforms the state-of-the-art on six text classification tasks, reducing the error by 18-24% on the majority of datasets. Furthermore, with only 100 labeled examples, it matches the performance of training from scratch on 100x more data. We open-source our pretrained models and code.</p>|

## 🛠️ Implementations

| Title | Description, Information |
| :---:         |          :--- |
|[Text transfer learning](https://docs.fast.ai/tutorial.text.html) by [fast.ai](https://docs.fast.ai/)|In this tutorial, we will see how we can train a model to classify text (here based on their sentiment). First we will see how to do this quickly in a few lines of code, then how to get state-of-the art results using the approach of the [ULMFit paper.](https://arxiv.org/abs/1801.06146)|
