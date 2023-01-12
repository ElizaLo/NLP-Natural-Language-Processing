# Keywords and Phrases Extraction

## Models

| Title | Description, Information |
| :---:         |          :--- |
| **GPT-3** | 📄 **Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165v4), [News Summarization and Evaluation in the Era of GPT-3](https://arxiv.org/pdf/2209.12356.pdf), [Papers with Code - GPT-3 Explained](https://paperswithcode.com/method/gpt-3)|

### 🔹 GPT-3: Generative Pre-trained Transformer

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

#### GPT-3 Tokenization

- [GPT-3 Online Tokenizer](https://beta.openai.com/tokenizer)

Tokenizer for GPT-3 is the same as GPT-2: 🤗 [OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#gpt2tokenizerfast).

A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).

The GPT-3 model can take as input from 4,000 to 2,000 tokens (**not confused with words!**). GPT-3 generates ~125-140% tokenks from the input text). 
> The text with 2,000 words approximately has 2,800 tokens.

Models understand and process text by breaking it down into tokens. Tokens can be words or just chunks of characters. For example, the word _**“hamburger”**_ gets broken up into the tokens _**“ham”**_, _**“bur”**_ and _**“ger”**_, while a short and common word like _**“pear”**_ is a single token. Many tokens start with a whitespace, for example _**“ hello”**_ and _**“ bye”**_.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/tokens.png" width="726" height="96"/>

> Common words like _**“cat”**_ are a single token, while less common words are often broken down into multiple tokens. _For example,_ _**“Butterscotch”**_ translates to four tokens: _**“But”**_, _**“ters”**_, _**“cot”**_, and _**“ch”**_.

The number of tokens processed in a given API request depends on the length of both your inputs and outputs. As a rough rule of thumb, 1 token is approximately 4 characters or 0.75 words for English text. One limitation to keep in mind is that your text prompt and generated completion combined must be no more than the model's maximum context length (for most models this is 2048 tokens, or about 1500 words). Check out [tokenizer tool](https://beta.openai.com/tokenizer) to learn more about how text translates to tokens.

#### `text-davinci-003` model

> OpenAI releases a new language model for GPT-3 trained with human feedback (**November 2022**). It brings numerous improvements, according to OpenAI.

The new GPT-3 model `“text-davinci-003”` is based on the [InstructGPT](https://the-decoder.com/openai-million-dollar-investment-and-a-new-ai-model/) models introduced by OpenAI earlier this year, which are optimized with human feedback. These models have already shown that AI models trained with RLHF (Reinforcement Learning from Human Feedback) can achieve better results with the same or even lower parameters.

According to OpenAI alignment researcher Jan Leike, `“text-davinci-003”` is largely equivalent to the InstructGPT models, but is not identical. The new model “scores higher on human preference ratings without being fundamentally more capable” than the underlying base model. For fine-tuning, OpenAI required “very little compute and data to align it compared to pretraining”.

Leike points out that the new GPT model still has “important limitations” and, for example, sometimes simply makes up things. However, such missteps should now “hopefully” be less frequent and less serious.

`“text-davinci-003”` can generate **“clearer, more engaging, and more compelling content”** and handle **more complex instructions**, according to OpenAI.

`“text-davinci-003”` can also write longer texts, according to OpenAI. As a result, the language AI can now take on tasks that were previously unfeasible. 

📰 **Articles:**

- [OpenAI’s latest GPT-3 model generates better and longer texts](https://the-decoder.com/openais-latest-gpt-3-model-generates-better-and-longer-texts/)

####  ⚙️ Fine-tuning model

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
4. Are you using `temperature`` and top_p` correctly?

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

- []()

#### 🔺 Evaluation Metrics

- NDCG (Normalized Discounted Cumulative Gain) 
- MRR (Mean Reciprocal Rank) 
- F1
- MAP (Mean Average Precision)

#### 💭 Conclusions

- The same `temperature` value generated different amounts of words and different words and phrases.
- This model is very sensitive to the input end, which influences the tokenization and consequently affects the extracted keywords.
    > **Warning:** Your text ends in a trailing space, which causes worse performance due to how the API splits text into tokens.
- It’s also worth taking note of the rather high `temperature` value and shorter max tokens. 
- Each time model generates text of different lengths. Setting up `max_tokens` doesn’t guarantee that text would be this length.
- Words in upper case are divided wrong into the tokens (e.g. FEBRUARY, CUISINE, RESEARCH, MENU, etc.). It is necessary to reduce all words to lowercase or to the form of ordinary sentences, where only the capital letter in the word is large.

#### 💡 Ideas

- Run the model with different `temperature` values and join results together to achieve more extracted keywords. 
- Run the model with different `temperature` values and count how many words are generated with each.
- Reduce all words to lowercase or to the form of ordinary sentences, where only the capital letter in the word is large.
- Clean texts to make them more structured and coherent.
- **Chunk/divide the input text:** divide the text into several smaller ones and make a keywords extraction for each of them separately and then combine it into one concatenated keywords list.
- Make chunking of documents, try and compare the results of summarization of **different GPT-3 models**: `text-davinci-003`, `text-curie-001`, `text-babbage-001` and `text-ada-001`.

#### ❓ Questions

- How to evaluate results?
- How to choose `temperature` value?
