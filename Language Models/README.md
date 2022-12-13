# Language Models

| Title | Description, Information |
| :---:         |          :--- |
| **GPT-3** | 📄 **Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165v4), [News Summarization and Evaluation in the Era of GPT-3](https://arxiv.org/pdf/2209.12356.pdf), [Papers with Code - GPT-3 Explained](https://paperswithcode.com/method/gpt-3)|

## 🔹 GPT-3: Generative Pre-trained Transformer

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

<img src="[https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/ezgif.com-gif-maker.jpg](https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/Language%20Models/img/ezgif.com-gif-maker.jpg)" width="1400" height="787">

### Zero Shot Text Summarization with GPT-3

Zero shot text summarization refers to using GPT-3 to summarize a given text input without providing any examples in the prompt. We simply provide the instructions for what we want GPT-3 to do and provide the text. 

The GPT-3 playground provides another example of summarization by simply adding a “tl;dr” to the end of the text passage. They consider this a “no instruction” example as they have not specified an initial task and rely entirely on the underlying language models' understanding of what “tl;dr” means.

**Zero Shot Summarization Explained**

Zero shot based GPT-3 prompts allow you to utilize the underlying model's understanding of your task given through instructions or headers in the prompt without being affected by examples. The model is only steered by any headers and instructions and it leverages these to grow its understanding of what you consider correct. At the end of the day, most GPT-3 tasks are somewhat relative. What you consider a correct response vs what the model considers correct vs what a client considers correct can all be somewhat different. In summarization, it can mean an emphasis on specific keywords, topics, or phrases. It can mean a specific length or contain specific proper nouns. 

Because we haven’t included any real information through examples or a more specific prompt you’re at the mercy of the models underlying understanding. This makes it more difficult to steer the model towards what we might consider a good summary. This is the key flaw in zero shot summarization, as it becomes much harder to fit our prompt to a test dataset as the variation grows. Language changes to just the header and instruction cause less and less change as a few things happen:
1. The size of the input text grows or becomes exponentially shorter. This also depends on the [GPT-3 engine](https://beta.openai.com/docs/engines/gpt-3)
2. The variance in the type or input text or origin of the input text grows. If we’ve fit a nice zero shot summarization model to paragraphs out of a textbook, then move to research papers we will most likely see a drop in accuracy.
‍
This makes zero shot summarizers relatively unstable and very hard to use in production without a full natural language processing pipeline. There are some use cases where it does make sense. 

**Large Document Zero Shot Summarization Problems**

You’ll need to split the input text into smaller chunks to pass through GPT-3. Should I just try to fill up as much of the prompt as I can with each run to minimize the total runs? One problem with this is that your model's ability to understand each sentence and its importance to the overall chunk will go down as the size grows. This doesn’t really affect extractive summarization as much you can simply just increase the sentence count, but abstractive will take a hit as it becomes harder to decide what information is valuable enough to fit into a summary as the “possible information pool” grows. You also limit the size of your summary that can be generated. 

Smaller chunks allow for more understanding per chunk but increase the risk of split contextual information. Let’s say you split a dialog or topic in half when chunking to summarize. If the contextual information from that dialog or topic is small or hard to decipher per chunk that model might not include it at all in the summary for either chunk. You’ve now taken an important part of the overall text and split the contextual information about it in half reducing the model's likelihood to consider it important. On the other side you might produce two summaries of the two chunks dominated by that dialog or topic.

### **Few Shot Summarization**

Few shot learning with GPT-3 refers to taking the underlying task agnostic large language model and showing the prompt actual examples of how to complete the task. The model combines its trained understanding of how to predict the next token in a language sequence and the “pattern” it picks up on in the prompt through examples to produce a much higher accuracy result. Accuracy is an odd idea here, as it really just follows the examples and tries to fit its quick learning to the new input. As you can imagine, if your examples are incorrect (sentiment analysis) or don't contain the output you would want, you’ll get a result that you don’t want. 

Few shot learning with relevant examples has been shown to **boost the accuracy of GPT-3 up to 30%** for some tasks, and the boost for summarization is no different. Relevant prompt examples help guide our GPT-3 summarizer to include specific language and topics in our summary without needing overly structured prompt instructions that can lead to overfitting. 

One key difference is our new ability to steer the model towards results that exactly fit what we want without needing multiple steps. We can use our examples to affect both extractive and abstractive type summarizations to account for what we want. 

One of the main differences between the two systems is how we set up the architecture for producing summaries. It’s no secret that GPT-3 performs better when the examples provided are relevant to the input text. Examples that discuss the same topics, are from the same news article or are closely related help GPT-3 better understand how to map input language to output language for our specific task. 

Most of full scale production summarization architectures are few shot learning based as seen them to produce the most flexibility and highest “accuracy” towards our goal. 

### GPT-3 Tokenization

- [GPT-3 Online Tokenizer](https://beta.openai.com/tokenizer)

Tokenizer for GPT-3 is the same as GPT-2: 🤗 [OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#gpt2tokenizerfast).

A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).

The GPT-3 model can take as input from 4,000 to 2,000 tokens (**not confused with words!**). GPT-3 generates ~125-140% tokenks from the input text). 
> The text with 2,000 words approximately has 2,800 tokens.

### `text-davinci-003` model

> OpenAI releases a new language model for GPT-3 trained with human feedback (**November 2022**). It brings numerous improvements, according to OpenAI.

The new GPT-3 model `“text-davinci-003”` is based on the [InstructGPT](https://the-decoder.com/openai-million-dollar-investment-and-a-new-ai-model/) models introduced by OpenAI earlier this year, which are optimized with human feedback. These models have already shown that AI models trained with RLHF (Reinforcement Learning from Human Feedback) can achieve better results with the same or even lower parameters.

According to OpenAI alignment researcher Jan Leike, `“text-davinci-003”` is largely equivalent to the InstructGPT models, but is not identical. The new model “scores higher on human preference ratings without being fundamentally more capable” than the underlying base model. For fine-tuning, OpenAI required “very little compute and data to align it compared to pretraining”.

Leike points out that the new GPT model still has “important limitations” and, for example, sometimes simply makes up things. However, such missteps should now “hopefully” be less frequent and less serious.

`“text-davinci-003”` can generate **“clearer, more engaging, and more compelling content”** and handle **more complex instructions**, according to OpenAI.

`“text-davinci-003”` can also write longer texts, according to OpenAI. As a result, the language AI can now take on tasks that were previously unfeasible.

#### 📰 Articles

- [OpenAI Released GPT-3 Text-davinci-003. I Compared It With 002. The Results Are Impressive!](https://pub.towardsai.net/openai-just-released-gpt-3-text-davinci-003-i-compared-it-with-002-the-results-are-impressive-dced9aed0cba)

### Other models from OpenAI based on GPT-3

Each of the GPT-3 models has its own USP. Pre-Davinci models such as **Curie**, **Babbage**, and **Ada** can do specific tasks very well at a faster rate and at a lower cost.

- **Curie** `text-curie-001` is suitable for classification and sentiment analysis tasks. The model also produces results for queries, and answers questions, and can be used as a general-purpose chatbot. The comparison shows that it can do many of the tasks of Davinci, but for 10% of the cost.

- **Babbage** `text-babbage-001` is best suited for simple classification tasks and performs SEO text analysis.

- **Ada** `text-ada-001`, the fastest of all models, is capable of tasks such as text parsing, address correction, and less complex classification tasks.

> **Другие модели от OpenAI на базе GPT-3**
> 
> Каждая из моделей GPT-3 имеет свое УТП. Модели, предшествовавшие **Davinci**, такие как **Curie**, **Babbage** и **Ada**, могут очень хорошо выполнять специфичные задачи с более высокой скоростью и за меньшую стоимость.
> 
> - **Curie** `text-curie-001` подходит для задач классификация и анализа настроений. Модель также выдает результаты на запросы, отвечает на вопросы и может использоваться в качестве чат-бота общего назначения. Сравнение показывает, что она может выполнять многие задачи Davinci, но за 10% стоимости.
> 
> - **Babbage** `text-babbage-001` лучше всего подходит для простых задач классификации и выполняет SEO-анализ текста.
> 
> - **Ada** `text-ada-001`, самая быстрая из всех моделей, способна выполнять такие задачи, как синтаксический анализ текста, исправление адреса и менее сложные задачи классификации.

📰 **Articles:**

- [OpenAI’s latest GPT-3 model generates better and longer texts](https://the-decoder.com/openais-latest-gpt-3-model-generates-better-and-longer-texts/)

### Fine-tuning model

### 💭 Conclusions

- The same temperature value generated different amounts of words and different words and phrases.
- This model is very sensitive to the input end, which influences the tokenization and consequently affects the generated text.
    > **Warning:** Your text ends in a trailing space, which causes worse performance due to how the API splits text into tokens.
- It’s also worth taking note of the rather high temperature value and shorter max tokens. 
- Each time model generates text of different lengths. Setting up `max_tokens` doesn’t guarantee that text would be this length.
- Words in upper case are divided wrong into the tokens (e.g. FEBRUARY, CUISINE, RESEARCH, MENU, etc.). It is necessary to reduce all words to lowercase or to the form of ordinary sentences, where only the capital letter in the word is large.

### 💡 Ideas
- Run the model with different `temperature` values and join results together to achieve more extracted keywords. 
- Run the model with different `temperature` values and count how many words are generated with each.
- - Reduce all words to lowercase or to the form of ordinary sentences, where only the capital letter in the word is large.
- Clean texts to make them more structured and coherent.
- ** Chunk/divide the input text:** divide the text into several smaller ones and make a keywords extraction for each of them separately and then combine it into one concatenated keywords list.
- Make chunking of documents, try and compare the results of summarization of **different GPT-3 models**: `text-davinci-003`, `text-curie-001`, `text-babbage-001` and `text-ada-001`.

### ❓ Questions

- How to evaluate results?
- How to choose `temperature` value?
