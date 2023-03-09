<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/img/Language_Models.png" width="1050" height="150"/>

| Title | Description, Information |
| :---:         |          :--- |
|**Universal Language Model Fine-tuning for Text Classification (ULMFiT)**|<p> 📄 **Paper:** [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) by Jeremy Howard, Sebastian Ruder</p><p>Inductive transfer learning has greatly impacted computer vision, but existing approaches in NLP still require task-specific modifications and training from scratch. We propose Universal Language Model Fine-tuning (ULMFiT), an effective transfer learning method that can be applied to any task in NLP, and introduce techniques that are key for fine-tuning a language model. Our method significantly outperforms the state-of-the-art on six text classification tasks, reducing the error by 18-24% on the majority of datasets. Furthermore, with only 100 labeled examples, it matches the performance of training from scratch on 100x more data. We open-source our pretrained models and code.</p>|
| **GPT-1** | 📄 **Paper:** [Papers with Code - Improving Language Understanding by Generative Pre-Training](https://paperswithcode.com/paper/improving-language-understanding-by)|
|**GPT-2**| 📄 **Paper:** [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)|
| **GPT-3** | 📄 **Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165v4), [News Summarization and Evaluation in the Era of GPT-3](https://arxiv.org/pdf/2209.12356.pdf), [Papers with Code - GPT-3 Explained](https://paperswithcode.com/method/gpt-3)|
|**Gopher**| 📄 **Paper:** |
|**Megatron**| 📄 **Paper:** |
|**InstructGPT**| 📄 **Paper:** [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)|
|**Pathways Language Model (PaLM)**| 📄 **Paper:** [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) |
|**Chinchilla**| 📄 **Paper:** [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556v1)|

# ♦️ Generative Pre-trained Transformer (GPT)

Before GPT-1, most Natural Language Processing (NLP) models were trained for particular tasks like classification, translation, etc. They all were using supervised learning. This type of learning comes with two issues: lack of annotated data and failure to generalize tasks.

- 📰 **Articles:**
  - [Everything We Know About GPT-4]([https://www.width.ai/post/gpt3-summarizer](https://www.datacamp.com/blog/what-we-know-gpt4)) - short history about all GPTs
  - [The Journey of Open AI GPT models](https://medium.com/walmartglobaltech/the-journey-of-open-ai-gpt-models-32d95b7b7fb2)

## 🔹 GPT-1: Generative Pre-trained Transformer

- 📄 **Paper:** [Papers with Code - Improving Language Understanding by Generative Pre-Training](https://paperswithcode.com/paper/improving-language-understanding-by)

GPT-1 (117M parameters) paper ([Improving Language Understanding by Generative Pre-Training](https://paperswithcode.com/paper/improving-language-understanding-by)) was published in 2018. It has proposed a generative language model that was trained on unlabeled data and fine-tuned on specific downstream tasks such as classification and sentiment analysis. 

## 🔹 GPT-2: Generative Pre-trained Transformer

- 📄 **Paper:** [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

GPT-2 (1.5B parameters) paper ([Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)) was published in 2019. It was trained on a larger dataset with more model parameters to build an even more powerful language model. GPT-2 uses task conditioning, Zero-Shot Learning, and Zero Short Task Transfer to improve model performance.

## 🔹 GPT-3: Generative Pre-trained Transformer

- 📄 **Paper:** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
  - 📃 Other related papers: [News Summarization and Evaluation in the Era of GPT-3](https://arxiv.org/abs/2209.12356)
  - [Papers with Code - GPT-3 Explained](https://paperswithcode.com/method/gpt-3)
- 🛠️ **Implementations:**
  - [OpenAI API - GPT-3 Documentation](https://beta.openai.com/docs/models/gpt-3), [OpenAI - Help, advice and answers from the OpenAI Team](https://help.openai.com/en/)
  > Read more - [Model index for researchers](https://beta.openai.com/docs/model-index-for-researchers)
  - [Fine-tuning](https://beta.openai.com/docs/guides/fine-tuning)
- **Modifications:**
  - :octocat: [minGPT](https://github.com/karpathy/minGPT) by **Andrej Karpathy**
  > A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training
  - :octocat: [nanoGPT](https://github.com/karpathy/nanoGPT) by **Andrej Karpathy**
    - :octocat: [nanogpt-lecture](https://github.com/karpathy/ng-video-lecture) 
  > The simplest, fastest repository for training/finetuning medium-sized GPTs.
- 📰 **Articles:**
  - [State of the Art GPT-3 Summarizer For Any Size Document or Format](https://www.width.ai/post/gpt3-summarizer)
  - [A Beginner's Guide to GPT-3](https://www.datacamp.com/blog/a-beginners-guide-to-gpt-3)
- :gear: **Notebook:** 
  - [GPT-3.ipynb](https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/Text%20Summarization/GPT-3.ipynb) - GPT-3 - Generative Pre-trained Transformer 3, **_model_**: `text-davinci-003` (released: _November 2022_)
  
### :octocat: GitHub Repositiories

| Title | Description, Information |
| :---:         |          :--- |
|[Gepetto](https://github.com/JusticeRage/Gepetto)|Gepetto is a Python script which uses OpenAI's `davinci-003` model to provide meaning to functions decompiled by IDA Pro. At the moment, it can ask davinci-003 to explain what a function does, and to automatically rename its variables.|

<img src="https://github.com/ElizaLo/NLP-Natural-Language-Processing/blob/master/Language%20Models/img/ezgif.com-gif-maker.jpg" width="700" height="394"/>

### 📚 Zero Shot Text Summarization with GPT-3

Zero shot text summarization refers to using GPT-3 to summarize a given text input without providing any examples in the prompt. We simply provide the instructions for what we want GPT-3 to do and provide the text. 

The GPT-3 playground provides another example of summarization by simply adding a “tl;dr” to the end of the text passage. They consider this a “no instruction” example as they have not specified an initial task and rely entirely on the underlying language models' understanding of what “tl;dr” means.

📘 **Zero Shot Summarization Explained**

Zero shot based GPT-3 prompts allow you to utilize the underlying model's understanding of your task given through instructions or headers in the prompt without being affected by examples. The model is only steered by any headers and instructions and it leverages these to grow its understanding of what you consider correct. At the end of the day, most GPT-3 tasks are somewhat relative. What you consider a correct response vs what the model considers correct vs what a client considers correct can all be somewhat different. In summarization, it can mean an emphasis on specific keywords, topics, or phrases. It can mean a specific length or contain specific proper nouns. 

Because we haven’t included any real information through examples or a more specific prompt you’re at the mercy of the models underlying understanding. This makes it more difficult to steer the model towards what we might consider a good summary. This is the key flaw in zero shot summarization, as it becomes much harder to fit our prompt to a test dataset as the variation grows. Language changes to just the header and instruction cause less and less change as a few things happen:
1. The size of the input text grows or becomes exponentially shorter. This also depends on the [GPT-3 engine](https://beta.openai.com/docs/engines/gpt-3)
2. The variance in the type or input text or origin of the input text grows. If we’ve fit a nice zero shot summarization model to paragraphs out of a textbook, then move to research papers we will most likely see a drop in accuracy.
‍
This makes zero shot summarizers relatively unstable and very hard to use in production without a full natural language processing pipeline. There are some use cases where it does make sense. 

📚 **Large Document Zero Shot Summarization Problems**

You’ll need to split the input text into smaller chunks to pass through GPT-3. Should I just try to fill up as much of the prompt as I can with each run to minimize the total runs? One problem with this is that your model's ability to understand each sentence and its importance to the overall chunk will go down as the size grows. This doesn’t really affect extractive summarization as much you can simply just increase the sentence count, but abstractive will take a hit as it becomes harder to decide what information is valuable enough to fit into a summary as the “possible information pool” grows. You also limit the size of your summary that can be generated. 

Smaller chunks allow for more understanding per chunk but increase the risk of split contextual information. Let’s say you split a dialog or topic in half when chunking to summarize. If the contextual information from that dialog or topic is small or hard to decipher per chunk that model might not include it at all in the summary for either chunk. You’ve now taken an important part of the overall text and split the contextual information about it in half reducing the model's likelihood to consider it important. On the other side you might produce two summaries of the two chunks dominated by that dialog or topic.

### 📗 **Few Shot Summarization**

GPT-3 has been pre-trained on a vast amount of text from the open internet. When given a prompt with just a few examples, it can often intuit what task you are trying to perform and generate a plausible completion. This is often called "few-shot learning."

Few shot learning with GPT-3 refers to taking the underlying task agnostic large language model and showing the prompt actual examples of how to complete the task. The model combines its trained understanding of how to predict the next token in a language sequence and the “pattern” it picks up on in the prompt through examples to produce a much higher accuracy result. Accuracy is an odd idea here, as it really just follows the examples and tries to fit its quick learning to the new input. As you can imagine, if your examples are incorrect (sentiment analysis) or don't contain the output you would want, you’ll get a result that you don’t want. 

Few shot learning with relevant examples has been shown to **boost the accuracy of GPT-3 up to 30%** for some tasks, and the boost for summarization is no different. Relevant prompt examples help guide our GPT-3 summarizer to include specific language and topics in our summary without needing overly structured prompt instructions that can lead to overfitting. 

One key difference is our new ability to steer the model towards results that exactly fit what we want without needing multiple steps. We can use our examples to affect both extractive and abstractive type summarizations to account for what we want. 

One of the main differences between the two systems is how we set up the architecture for producing summaries. It’s no secret that GPT-3 performs better when the examples provided are relevant to the input text. Examples that discuss the same topics, are from the same news article or are closely related help GPT-3 better understand how to map input language to output language for our specific task. 

Most of full scale production summarization architectures are few shot learning based as seen them to produce the most flexibility and highest “accuracy” towards our goal. 

### 🔡 GPT-3 Tokenization

- [GPT-3 Online Tokenizer](https://beta.openai.com/tokenizer)

Tokenizer for GPT-3 is the same as GPT-2: 🤗 [OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#gpt2tokenizerfast).

A helpful rule of thumb is that one token generally corresponds to ~4 characters of text for common English text. This translates to roughly ¾ of a word (so 100 tokens ~= 75 words).

The GPT-3 model can take as input from 4,000 to 2,000 tokens (**not confused with words!**). GPT-3 generates ~125-140% tokenks from the input text). 
> The text with 2,000 words approximately has 2,800 tokens.

Models understand and process text by breaking it down into tokens. Tokens can be words or just chunks of characters. For example, the word _**“hamburger”**_ gets broken up into the tokens _**“ham”**_, _**“bur”**_ and _**“ger”**_, while a short and common word like _**“pear”**_ is a single token. Many tokens start with a whitespace, for example _**“ hello”**_ and _**“ bye”**_.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/tokens.png" width="726" height="96"/>

> Common words like _**“cat”**_ are a single token, while less common words are often broken down into multiple tokens. _For example,_ _**“Butterscotch”**_ translates to four tokens: _**“But”**_, _**“ters”**_, _**“cot”**_, and _**“ch”**_.

The number of tokens processed in a given API request depends on the length of both your inputs and outputs. As a rough rule of thumb, 1 token is approximately 4 characters or 0.75 words for English text. One limitation to keep in mind is that your text prompt and generated completion combined must be no more than the model's maximum context length (for most models this is 2048 tokens, or about 1500 words). Check out [tokenizer tool](https://beta.openai.com/tokenizer) to learn more about how text translates to tokens.

### 🔑 Key concepts for GPT-3 by OpenAI

> Our models are used for both research purposes and developer use cases in production. Researchers often learn about our models from papers that we have published, but there is often not a perfect match between what is available in the OpenAI API and what is published in a paper.
>
> The purpose of this page is to help clarify:
>
> - Some of the differences in the ways that our models are trained, which impacts the comparisons that can be made between models, and various evaluation results.
> - The differences between various model series, such as GPT 3.5 and InstructGPT.
> - Which if any of the models available in the API today match with a model in a paper. In some cases, there might not be a match.

The [completions](https://beta.openai.com/docs/api-reference/completions) endpoint is at the center of our API. It provides a simple interface to our models that is extremely flexible and powerful. _You input some text as a **prompt**, and the model will generate a text **completion** that attempts to match whatever context or pattern you gave it._ 

For example, if you give the API the prompt, _“Write a tagline for an ice cream shop”_, it will return a completion like _“We serve up smiles with every scoop!”_

[Designing your prompt](https://beta.openai.com/docs/guides/completion/prompt-design) is essentially how you “program” the model, usually by providing some instructions or a few examples. This is different from most other NLP services which are designed for a single task, such as sentiment classification or named entity recognition. Instead, the completions endpoint can be used for virtually any task including content or code generation, summarization, expansion, conversation, creative writing, style transfer, and more.

### Models referred to as "GPT 3.5"

> Read more - [Model index for researchers](https://beta.openai.com/docs/model-index-for-researchers)

GPT-3.5 series is a series of models that was trained on a blend of text and code from before Q4 2021. The following models are in the GPT-3.5 series:

1. `code-davinci-002` is a base model, so good for pure code-completion tasks
2. `text-davinci-002` is an InstructGPT model based on `code-davinci-002`
3. `text-davinci-003` is an improvement on `text-davinci-002`
4. `gpt-3.5-turbo-0301` is an improvement on `text-davinci-003`, optimized for chat

### 🔸 Turbo

Turbo is the same model family that powers ChatGPT. It is optimized for conversational chat input and output but does equally well on completions when compared with the Davinci model family. Any use case that can be done well in ChatGPT should perform well with the Turbo model family in the API.

The Turbo model family is also the first to receive regular model updates like ChatGPT.

- ➡️ Good at: **Conversation** and **text generation**

Using the OpenAI API, you can build your own applications with gpt-3.5-turbo to do things like:

- Draft an email or other piece of writing
- Write Python code
- Answer questions about a set of documents
- Create conversational agents
- Give your software a natural language interface
- Tutor in a range of subjects
- Translate languages
- Simulate characters for video games and much more

Chat models take a series of messages as input, and return a model-generated message as output.

Although the chat format is designed to make multi-turn conversations easy, it’s just as useful for single-turn tasks without any conversations (such as those previously served by instruction following models like `text-davinci-003`).

An example API call looks as follows:

```python
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)
```

The main input is the messages parameter. Messages must be an array of message objects, where each object has a role (either "system", "user", or "assistant") and content (the content of the message). Conversations can be as short as 1 message or fill many pages.

Typically, a conversation is formatted with a system message first, followed by alternating user and assistant messages.

The system message helps set the behavior of the assistant. In the example above, the assistant was instructed with "You are a helpful assistant."

The user messages help instruct the assistant. They can be generated by the end users of an application, or set by a developer as an instruction.

The assistant messages help store prior responses. They can also be written by a developer to help give examples of desired behavior.

Including the conversation history helps when user instructions refer to prior messages. In the example above, the user’s final question of "Where was it played?" only makes sense in the context of the prior messages about the World Series of 2020. Because the models have no memory of past requests, all relevant information must be supplied via the conversation. If a conversation cannot fit within the model’s token limit, it will need to be shortened in some way.

▶️ **Response format**

An example API response looks as follows:

```python
{
 'id': 'chatcmpl-6p9XYPYSTTRi0xEviKjjilqrWU2Ve',
 'object': 'chat.completion',
 'created': 1677649420,
 'model': 'gpt-3.5-turbo',
 'usage': {'prompt_tokens': 56, 'completion_tokens': 31, 'total_tokens': 87},
 'choices': [
   {
    'message': {
      'role': 'assistant',
      'content': 'The 2020 World Series was played in Arlington, Texas at the Globe Life Field, which was the new home stadium for the Texas Rangers.'},
    'finish_reason': 'stop',
    'index': 0
   }
  ]
}
```

In Python, the assistant’s reply can be extracted with `response['choices'][0]['message']['content']`.

Every response will include a `finish_reason`. The possible values for `finish_reason` are:

- `stop`: API returned complete model output
- `length`: Incomplete model output due to [`max_tokens` parameter](https://platform.openai.com/docs/api-reference/chat/create#chat/create-max_tokens) or token limit
- `content_filter`: Omitted content due to a flag from our content filters
- `null`: API response still in progress or incomplete

#### 📄 Documentation:

- [Models - Turbo - OpenAI API](https://platform.openai.com/docs/models/turbo)
- [Chat completions - OpenAI API](https://platform.openai.com/docs/guides/chat)
- [Create chat completion - OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat/create)
- ⚠️ [AttributeError: module ‘openai’ has no attribute ‘ChatCompletion’](https://community.openai.com/t/attributeerror-module-openai-has-no-attribute-chatcompletion/81490)
- :octocat: [openai-python/chatml.md at main · openai/openai-python](https://github.com/openai/openai-python/blob/main/chatml.md)

```python
!pip show openai
```

**Output:**

```python
Name: openai
Version: 0.27.0
Summary: Python client library for the OpenAI API
Home-page: https://github.com/openai/openai-python
Author: OpenAI
Author-email: support@openai.com
License: 
Location: /opt/conda/lib/python3.9/site-packages
Requires: aiohttp, requests, tqdm
Required-by: 
```

#### 🔡 Tokenization

To see how many tokens are used by an API call, check the usage field in the API response (e.g., `response['usage']['total_tokens']`).

Chat models like `gpt-3.5-turbo` use tokens in the same way as other models, but because of their message-based formatting, it's more difficult to count how many tokens will be used by a conversation.

**Counting tokens for chat API calls**

Below is an example function for counting tokens for messages passed to `gpt-3.5-turbo-0301`.
The exact way that messages are converted into tokens may change from model to model. So when future model versions are released, the answers returned by this function may be only approximate. The [ChatML documentation](https://github.com/openai/openai-python/blob/main/chatml.md) explains how messages are converted into tokens by the OpenAI API, and may be useful for writing your own function.

To see how many tokens are in a text string without making an API call, use OpenAI’s [tiktoken](https://github.com/openai/tiktoken) Python library. Example code can be found in the OpenAI Cookbook’s guide on [how to count tokens with tiktoken](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb).

Each message passed to the API consumes the number of tokens in the content, role, and other fields, plus a few extra for behind-the-scenes formatting. This may change slightly in the future.

If a conversation has too many tokens to fit within a model’s maximum limit _(e.g., more than **4096** tokens for `gpt-3.5-turbo`)_, you will have to truncate, omit, or otherwise shrink your text until it fits. Beware that if a message is removed from the messages input, the model will lose all knowledge of it.

Note too that very long conversations are more likely to receive incomplete replies. _For example,_ a `gpt-3.5-turbo` conversation that is 4090 tokens long will have its reply cut off after just 6 tokens.

#### Instructing chat models

Best practices for instructing models may change from model version to version. The advice that follows applies to gpt-3.5-turbo-0301 and may not apply to future models.

Many conversations begin with a system message to gently instruct the assistant. For example, here is one of the system messages used for ChatGPT:

`You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}`

In general, `gpt-3.5-turbo-0301` does not pay strong attention to the system message, and therefore important instructions are often better placed in a user message.

If the model isn’t generating the output you want, feel free to iterate and experiment with potential improvements. You can try approaches like:

- Make your instruction more explicit
- Specify the format you want the answer in
- Ask the model to think step by step or debate pros and cons before settling on an answer

For more prompt engineering ideas, read the OpenAI Cookbook guide on [techniques to improve reliability](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md).

Beyond the system message, the temperature and max tokens are [two of many options](https://platform.openai.com/docs/api-reference/chat) developers have to influence the output of the chat models. For temperature, higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. In the case of max tokens, if you want to limit a response to a certain length, max tokens can be set to an arbitrary number. This may cause issues for example if you set the max tokens value to 5 since the output will be cut-off and the result will not make sense to users.

#### Chat vs Completions

Because `gpt-3.5-turbo` performs at a similar capability to `text-davinci-003` but at 10% the price per token, we recommend `gpt-3.5-turbo` for most use cases.

For many developers, the transition is as simple as rewriting and retesting a prompt.

For example, if you translated English to French with the following completions prompt:

```python
Translate the following English text to French: "{text}"
```

An equivalent chat conversation could look like:

```python
[
  {"role": "system", "content": "You are a helpful assistant that translates English to French."},
  {"role": "user", "content": 'Translate the following English text to French: "{text}"'}
]
```

Or even just the user message:

```python
[
  {"role": "user", "content": 'Translate the following English text to French: "{text}"'}
]
```


#### ❓FAQ

- Is fine-tuning available for gpt-3.5-turbo?

No. As of Mar 1, 2023, you can only fine-tune base GPT-3 models. See the fine-tuning guide for more details on how to use fine-tuned models.

#### 🛠️ Request body

- `model` _(string, **Required**)_ — ID of the model to use. Currently, only `gpt-3.5-turbo` and `gpt-3.5-turbo-0301` are supported.
- `messages` _(array, **Required**)_ — The messages to generate chat completions for, in the [chat format](https://platform.openai.com/docs/guides/chat/introduction).
- `temperature` _(number, Optional, Defaults to 1)_ — What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both.
- `top_p` _(number, Optional, Defaults to 1)_ — An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.
- `n` _(integer, Optional, Defaults to 1)_ — How many chat completion choices to generate for each input message.
- `stream` _(boolean, Optional, Defaults to false)_ - If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message.
- `stop` _(string or array, Optional, Defaults to null)_ - Up to 4 sequences where the API will stop generating further tokens.
- `max_tokens` _(integer, Optional, Defaults to inf)_ - The maximum number of tokens allowed for the generated answer. By default, the number of tokens the model can return will be (4096 - prompt tokens).
- `presence_penalty` _(number, Optional, Defaults to 0)_ - Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
- `frequency_penalty` _(number, Optional, Defaults to 0)_ - Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. [See more information about frequency and presence penalties.](https://platform.openai.com/docs/api-reference/parameter-details)
- `logit_bias` _(map, Optional, Defaults to null)_ - Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
- `user` _(string, Optional)_ - A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more.](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids)

> ⚠️ **Note:** you need to be using OpenAI Python `v0.27.0` for the code below to work

```python
openai.ChatCompletion.create(
  model = "gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Extract keywords and key phrases from this text and divide by comma:\n\n{}\n\n".format(text)}
  ],
  temperature = 0.7,
  #top_p = 1.0,
  n = 3,
  max_tokens = 600,
  #stream = True,
  #stop = None,
  presence_penalty = 1,  #0.0
  frequency_penalty = 1, #0.8
  #logit_bias = {} #remove word
  #user = 'user_1'   
)
```

### 🔸 Davinci

Davinci is the most capable model family and can perform any task the other models can perform and often with less instruction. For applications requiring a lot of understanding of the content, like summarization for a specific audience and creative content generation, Davinci is going to produce the best results.

Another area where Davinci shines is in understanding the intent of text. Davinci is quite good at solving many kinds of logic problems and explaining the motives of characters. Davinci has been able to solve some of the most challenging AI problems involving cause and effect.

- ➡️ Good at: **Complex intent, cause and effect, summarization for audience**

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

The API is powered by a set of models with different capabilities and price points. **Codex** series is a descendant of GPT-3 that’s been trained on both natural language and code.

Each of the GPT-3 models has its own USP. Pre-Davinci models such as **Curie**, **Babbage**, and **Ada** can do specific tasks very well at a faster rate and at a lower cost.

> Каждая из моделей GPT-3 имеет свое УТП. Модели, предшествовавшие **Davinci**, такие как **Curie**, **Babbage** и **Ada**, могут очень хорошо выполнять специфичные задачи с более высокой скоростью и за меньшую стоимость.

### 🔸 Curie

Curie is extremely powerful, yet very fast. While Davinci is stronger when it comes to analyzing complicated text, Curie is quite capable for many nuanced tasks like sentiment classification and summarization. Curie is also quite good at answering questions and performing Q&A and as a general service chatbot.

While Davinci is more capable when it comes to comprehending text and generating responses that are more nuanced like summarizing for a child or emulating human speaking patterns, Curie is highly capable of analyzing text, answering direct questions, and providing key points.

Curie `text-curie-001` is suitable for classification and sentiment analysis tasks. The model also produces results for queries, and answers questions, and can be used as a general-purpose chatbot. The comparison shows that it can do many of the tasks of Davinci, but for 10% of the cost.

> Curie `text-curie-001` подходит для задач классификация и анализа настроений. Модель также выдает результаты на запросы, отвечает на вопросы и может использоваться в качестве чат-бота общего назначения. Сравнение показывает, что она может выполнять многие задачи Davinci, но за 10% стоимости.

Curie is highly capable of getting important information from text and very useful for a variety of applications including:
- Turning technical documents into bullet points
- Extracting important information from email
- Getting key points from customer service communication


- ➡️ Good at: **Language translation, complex classification, text sentiment, summarization**

### 🔸 Babbage

Babbage can perform straightforward tasks like simple classification. It’s also quite capable when it comes to Semantic Search ranking how well documents match up with search queries.

Babbage `text-babbage-001` is best suited for simple classification tasks and performs SEO text analysis.

> Babbage `text-babbage-001` лучше всего подходит для простых задач классификации и выполняет SEO-анализ текста.

Babbage is good at picking up obvious patterns in text and then using that as a reference to generate text. Babbage can also perform broad classification tasks like assigning categories to industries, genres and media content. For creative applications, Babbage is able to understand enough structure to create simple plots and titles.

- **Idea iteration**

You can give Babbage a prompt as simple as “Provide 7 tips for better YouTube videos,” and it will automatically create a list of practical advice. You can do this for just about any topic that is reasonably well known. 

- **Sentence completion**

Babbage can work as a great brainstorming tool and help someone complete their thoughts. If you start a paragraph or sentence, Babbage can quickly get the context and serve as a writing assistant.

- **Plot generation**

If you provide Babbage with a list of plot ideas from a specific genre, it can continue adding to that list. If you select the good ones and delete the others, you can keep sending the growing list to the API and improve the results.

- ➡️ Good at: **Moderate classification, semantic search classification**

### 🔸 Ada

Ada is usually the fastest model and can perform tasks like parsing text, address correction and certain kinds of classification tasks that don’t require too much nuance. Ada’s performance can often be improved by providing more context.

Ada `text-ada-001`, the fastest of all models, is capable of tasks such as text parsing, address correction, and less complex classification tasks.

> Ada `text-ada-001`, самая быстрая из всех моделей, способна выполнять такие задачи, как синтаксический анализ текста, исправление адреса и менее сложные задачи классификации.

Ada is extremely fast and capable when it comes to tasks where creativity is more important than precision. This can be very useful for creative applications and for generating large datasets.

- **Random data**

Ada can quickly generate large amounts of data like names and addresses to be used for experimenting, building machine models and testing applications.

- **Character descriptions**

You can use Ada to create character descriptions by sending a handful of examples to the API. By adjusting the temperature and repetition settings you can control the randomness of the generated examples.

- ➡️ Good at: **Parsing text, simple classification, address correction, keywords**

Davinci is the most capable model, and Ada is the fastest.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/OpenAI_models_4_1.png" width="776" height="543"/>

While Davinci is generally the most capable, the other models can perform certain tasks extremely well with significant speed or cost advantages. For example, Curie can perform many of the same tasks as Davinci, but faster and for 1/10th the cost.

### 🔸 Codex

The Codex models are descendants of GPT-3 models that can understand and generate code. Their training data contains both natural language and billions of lines of public code from GitHub. Learn more.

They’re most capable in Python and proficient in over a dozen languages including **JavaScript**, **Go**, **Perl**, **PHP**, **Ruby**, **Swift**, **TypeScript**, **SQL**, and even **Shell**.

Currently offered two Codex models:

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/OpenAI_models_5.png" width="790" height="284"/>

To learn more, visit [models documentation](https://beta.openai.com/docs/models).

📰 **Articles:**

- [OpenAI’s latest GPT-3 model generates better and longer texts](https://the-decoder.com/openais-latest-gpt-3-model-generates-better-and-longer-texts/)

### 💠 Text and Topic Segmentation / Chunking of long texts (longer than `4097`)

> Read more about segmentation of long text in 📂 [Topic and Text Segmentation](https://github.com/ElizaLo/NLP-Natural-Language-Processing/tree/master/Topic%20and%20Text%20Segmentation) folder

Chunking can be made with Fast Tokenizers from HuggingFace ([Fast tokenizers in the QA pipeline - Hugging Face Course](https://huggingface.co/course/chapter6/3b?fw=pt#handling-long-contexts)) with GPT-2 tokenizer fast ([OpenAI GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2TokenizerFast)) since GPT-3 and GPT-2 has the same tokenizer.

```python

from transformers import GPT2TokenizerFast


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

inputs = tokenizer(
    long_context,
    stride=50,
    max_length=3400,
    truncation=True,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

tokenizer.decode(inputs["input_ids"][0])[:50]
```

**How can I tell how many tokens a string will have before I embed it?**

For second-generation embedding models, as of Dec 2022, there is not yet a way to count tokens locally. The only way to get total token counts is to submit an API request.

- If the request succeeds, you can extract the number of tokens from the response: `response[“usage”][“total_tokens”]`
- If the request fails for having too many tokens, you can extract the number of tokens from the error message: _e.g._, `This model's maximum context length is 8191 tokens, however you requested 10000 tokens (10000 in your prompt; 0 for the completion). Please reduce your prompt; or completion length.`

For first-generation embedding models, which are based on GPT-2/GPT-3 tokenization, you can count tokens in a few ways:

- For one-off checks, the [OpenAI tokenizer](https://beta.openai.com/tokenizer) page is convenient
- In Python, [transformers.GPT2TokenizerFast](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2TokenizerFast) (the GPT-2 tokenizer is the same as GPT-3)
- In JavaScript, [gpt-3-encoder](https://www.npmjs.com/package/gpt-3-encoder)

Python example:

```python
from transformers import GPT2TokenizerFast

def num_tokens_from_string(string: str, tokenizer) -> int:
    return len(tokenizer.encode(string))

string = "your text here"
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

num_tokens_from_string(string, tokenizer)
```

- 💡 There is another possible improvement of chunking with the SBERT model ([SentenceTransformers Documentation — Sentence-Transformers  documentation](https://www.sbert.net)) - 📰 [How to chunk text into paragraphs using python](https://medium.com/@npolovinkin/how-to-chunk-text-into-paragraphs-using-python-8ae66be38ea6).

### ⚙️ Fine-tuning model

🔢 **Fine-tuning parameters in the GPT-3 model**

➡️ **Parameters:**

- `model` _(string, Required)_ – ID of the model to use. You can use the [List models](https://beta.openai.com/docs/api-reference/models/list) API to see all of your available models, or see our [Model overview](https://beta.openai.com/docs/models/overview) for descriptions of them.
- `prompt` _(string or array, Optional, Defaults to `<|endoftext|>`)_ – The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
  > **Note** that `<|endoftext|>` is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document.
- `suffix` _(string, Optional, Defaults to null)_ – The suffix that comes after a completion of inserted text.
- `max_tokens` _(integer, Optional, Defaults to 16)_ – The maximum number of [tokens](https://beta.openai.com/tokenizer) to generate in the completion. The token count of your prompt plus `max_tokens` cannot exceed the model's context length. Most models have a context length of **2048** tokens (except for the newest models, which support **4096**).
- `temperature` _(number, Optional, Defaults to 1)_ – What [sampling temperature](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277) to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer. ‼️ We generally recommend altering this or `top_p` but not both.
- `top_p` _(number, Optional, Defaults to 1)_ – An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. ‼️ We generally recommend altering this or `temperature` but not both.
- `n` _(integer, Optional, Defaults to 1)_ – How many completions to generate for each prompt. **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and stop.
- stream (boolean, Optional, Defaults to false) – Whether to stream back partial progress. If set, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message.
- `logprobs` _(integer, Optional, Defaults to null)_ – Include the log probabilities on the `logprobs` most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens. The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response. The maximum value for `logprobs` is 5. If you need more than this, please contact us through our [Help center](https://help.openai.com/) and describe your use case.
- `echo` _(boolean, Optional, Defaults to false)_ – Echo back the prompt in addition to the completion
- `stop` _(string or array, Optional, Defaults to null)_ – Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
- `presence_penalty` _(number, Optional, Defaults to 0)_ – Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. [See more information about frequency and presence penalties](https://beta.openai.com/docs/api-reference/parameter-details).
- `frequency_penalty` _(number, Optional, Defaults to 0)_ – Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. See more information about frequency and presence penalties.
- `best_of` _(integer, Optional, Defaults to 1)_ – Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed. When used with `n`, `best_of` controls the number of candidate completions and `n` specifies how many to return – `best_of` must be greater than `n`. **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`.
- `logit_bias` _(map, Optional, Defaults to null)_ – Modify the likelihood of specified tokens appearing in the completion. Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](https://beta.openai.com/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token. **_As an example,_** you can pass `{"50256": -100}` to prevent the `<|endoftext|>` token from being generated.
- `user` _(string, Optional)_ – A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](https://beta.openai.com/docs/guides/safety-best-practices/end-user-ids).

#### 📰 Articles

- [Why am I getting different completions on Playground vs. the API?](https://help.openai.com/en/articles/6643200-why-am-i-getting-different-completions-on-playground-vs-the-api)
- [Controlling the length of Completions](https://help.openai.com/en/articles/5072518-controlling-the-length-of-completions)
- [Using logit bias to define token probability](https://help.openai.com/en/articles/5247780-using-logit-bias-to-define-token-probability)
- [Controlling GPT-3 with Logit Bias](https://aidungeon.medium.com/controlling-gpt-3-with-logit-bias-55866d593292)

###  🌡️ **Temperature**

📰 [How to sample from language models](https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277)

Remember that the model predicts which text is most likely to follow the text preceding it. Temperature is a value between 0 and 1 that essentially lets you control how confident the model should be when making these predictions. Lowering temperature means it will take fewer risks, and completions will be more accurate and deterministic. Increasing temperature will result in more diverse completions.

Given some text, the model determines which token is most likely to come next. _For example,_ the text _**“Horses are my favorite”**_ is most likely to be followed with the token _**“ animal”**_.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/probabilities.png" width="737" height="201"/>

This is where temperature comes into play. If you submit this prompt 4 times with temperature set to 0, the model will always return _**“ animal”**_ next because it has the highest probability. If you increase the temperature, it will take more risks and consider tokens with lower probabilities.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/temperature.png" width="713" height="201"/>

It’s usually best to set a low temperature for tasks where the desired output is well-defined. Higher temperature may be useful for tasks where variety or creativity are desired, or if you'd like to generate a few variations for your end users or human experts to choose from.

**Use a low temperature when extracting data**

_For example,_ we’ve set the temperature low because we’re looking for straight-forward answers to questions that the customer comment provides. We’re not asking the model to try to be creative with its responses – especially for yes or no questions.

‼️ The actual completion you see may differ because the API is stochastic by default. This means that you might get a slightly different completion every time you call it, even if your prompt stays the same. You can control this behavior with the [temperature](https://beta.openai.com/docs/api-reference/completions/create#completions/create-temperature) setting.

------

### 🔃 Training

Fine-tuning lets you get more out of the models available through the API by providing:

1. Higher quality results than prompt design
2. Ability to train on more examples than can fit in a prompt
3. Token savings due to shorter prompts
4. Lower latency requests

Fine-tuning improves on few-shot learning by training on many more examples than can fit in the prompt, letting you achieve better results on a wide number of tasks. **Once a model has been fine-tuned, you won't need to provide examples in the prompt anymore.** This saves costs and enables lower-latency requests.

At a high level, fine-tuning involves the following steps:

1. Prepare and upload training data
2. Train a new fine-tuned model
3. Use your fine-tuned model

Data must be a [JSONL](https://jsonlines.org/) document, where each line is a prompt-completion pair corresponding to a training example. You can use [CLI data preparation tool](https://beta.openai.com/docs/guides/fine-tuning/cli-data-preparation-tool) to easily convert your data into this file format.

``` Jsonl
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
...
```
Designing your prompts and completions for fine-tuning is different from designing your prompts for use with our base models (Davinci, Curie, Babbage, Ada). In particular, while prompts for base models often consist of multiple examples ("few-shot learning"), for fine-tuning, each training example generally consists of a single input example and its associated output, without the need to give detailed instructions or include multiple examples in the same prompt.

The more training examples you have, the better. **It's recommend having at least a couple hundred examples. In general, were found that each doubling of the dataset size leads to a linear increase in model quality.**

Every fine-tuning job starts from a base model, which defaults to **curie**. The choice of model influences both the performance of the model and the cost of running your fine-tuned model. Your model can be one of: `ada`, `babbage`, `curie`, or `davinci`.

You may continue to use all the other [Completions](https://beta.openai.com/docs/api-reference/completions) parameters like `temperature`, `frequency_penalty`, `presence_penalty`, etc, on these requests to fine-tuned models.

💠 **Preparing your dataset**

**Data formatting**

To fine-tune a model, you'll need a set of training examples that each consist of a single input ("prompt") and its associated output ("completion"). This is notably different from using our base models, where you might input detailed instructions or multiple examples in a single prompt.

- Each prompt should end with a fixed separator to inform the model when the prompt ends and the completion begins. A simple separator which generally works well is `\n\n###\n\n`. The separator should not appear elsewhere in any prompt.
- Each completion should start with a whitespace due to our [tokenization](https://beta.openai.com/tokenizer), which tokenizes most words with a preceding whitespace.
- Each completion should end with a fixed stop sequence to inform the model when the completion ends. A stop sequence could be `\n`, `###`, or any other token that does not appear in any completion.
- For inference, you should format your prompts in the same way as you did when creating the training dataset, including the same separator. Also specify the same stop sequence to properly truncate the completion.

💠 **General best practices**

Fine-tuning performs better with more high-quality examples. To fine-tune a model that performs better than using a high-quality prompt with our base models, you should provide at least a few hundred high-quality examples, ideally vetted by human experts. From there, performance tends to linearly increase with every doubling of the number of examples. Increasing the number of examples is usually the best and most reliable way of improving performance.

**Classifiers** are the easiest models to get started with. For classification problems we suggest using `ada`, which generally tends to perform only very slightly worse than more capable models once fine-tuned, whilst being significantly faster and cheaper.

If you are fine-tuning on a pre-existing dataset rather than writing prompts from scratch, be sure to manually review your data for offensive or inaccurate content if possible, or review as many random samples of the dataset as possible if it is large.

----

- 🔸 **Change prompt**

_For example:_

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

📰 **Articles**

- [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)

-----

### 🔢 **Hyperparameters**

That said, tweaking the hyperparameters used for fine-tuning can often lead to a model that produces higher quality output. In particular, you may want to configure the following:

- `model:` The name of the base model to fine-tune. You can select one of "ada", "babbage", "curie", or "davinci". To learn more about these models, see the [Models](https://beta.openai.com/docs/models) documentation.
- `n_epochs` - defaults to 4. The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset.
- `batch_size` - defaults to ~0.2% of the number of examples in the training set, capped at 256. The batch size is the number of training examples used to train a single forward and backward pass. In general, we've found that larger batch sizes tend to work better for larger datasets.
- `learning_rate_multiplier` - defaults to 0.05, 0.1, or 0.2 depending on final `batch_size`. The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this multiplier. We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results. Empirically, we've found that larger learning rates often perform better with larger batch sizes.
- `compute_classification_metrics` - defaults to False. If True, for fine-tuning for classification tasks, computes classification-specific metrics (accuracy, F-1 score, etc) on the validation set at the end of every epoch.

### Example notebooks

- [Example notebooks - OpenAI API](https://beta.openai.com/docs/guides/fine-tuning/example-notebooks) :
  - :octocat: [openai-cookbook/examples/fine-tuned_qa at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/tree/main/examples/fine-tuned_qa) 

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
- Reduce all words to lowercase or to the form of ordinary sentences, where only the capital letter in the word is large.
- Clean texts to make them more structured and coherent.
- **Chunk/divide the input text:** divide the text into several smaller ones and make a keywords extraction for each of them separately and then combine it into one concatenated keywords list.
- Make chunking of documents, try and compare the results of summarization of **different GPT-3 models**: `text-davinci-003`, `text-curie-001`, `text-babbage-001` and `text-ada-001`.

### ❓ Questions

- How to evaluate results?
- How to choose `temperature` value?

## 🔹 InstructGPT

- 📄 **Paper:** [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- 🛠️ **Implementations:**
  - :octocat: [InstructGPT: Training Language Models to Follow Instructions with Human Feedback](https://github.com/openai/following-instructions-human-feedback)
- 📰 **Articles:**
  - [Aligning Language Models to Follow Instructions](https://openai.com/blog/instruction-following/) by OpenAI
- :gear: **Notebook:** 

### Models referred to as "GPT 3.5" in OpenAI

> Read more - [Model index for researchers](https://beta.openai.com/docs/model-index-for-researchers)

GPT-3.5 series is a series of models that was trained on a blend of text and code from before Q4 2021. The following models are in the GPT-3.5 series:

1. `code-davinci-002` is a base model, so good for pure code-completion tasks
2. `text-davinci-002` is an InstructGPT model based on `code-davinci-002`
3. `text-davinci-003` is an improvement on `text-davinci-002`

### InstructGPT models in OpenAI (January 2022)

We offer variants of InstructGPT models trained in 3 different ways:

| TRAINING METHOD | MODELS |
| :---        |          :--- |
|<p>**SFT**</p><p>Supervised fine-tuning on human demonstrations</p>|`davinci-instruct-beta`|
|<p>**FeedME**</p><p>Supervised fine-tuning on human-written demonstrations and on model samples rated 7/7 by human labelers on an overall quality score</p>|`text-davinci-001`, `text-davinci-002`, `text-curie-001`, `text-babbage-001`|
|<p>**PPO**</p><p>Reinforcement learning with reward models trained from comparisons by humans</p>|`text-davinci-003`|

The SFT and PPO models are trained similarly to the ones from the 📄 [InstructGPT](https://arxiv.org/abs/2203.02155) paper. FeedME (short for "feedback made easy") models are trained by distilling the best completions from all of our models. Our models generally used the best available datasets at the time of training, and so different engines using the same training methodology might be trained on different data.

## 🔹 ChatGPT

- 📄 **Paper:** 
- 🛠️ **Implementations:**
- 📰 **Articles:**
  - :hugs: [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) by HuggingFace
  - [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/) by OpenAI
- :gear: **Notebook:** 
  
### :octocat: GitHub Repositiories

| Title | Description, Information |
| :---:         |          :--- |
|[Awesome ChatGPT](https://github.com/humanloop/awesome-chatgpt)|Curated list of awesome tools, demos, docs for ChatGPT and GPT-3|
|[ChatGPT](https://github.com/acheong08/ChatGPT)|Lightweight package for interacting with ChatGPT's API by OpenAI. Uses reverse engineered official API.|
|[]()| |

## 💠 Models by OpenAI

The OpenAI API is powered by a family of models with different capabilities and price points. You can also customize our base models for your specific use case with [fine-tuning](https://beta.openai.com/docs/guides/fine-tuning).

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/OpenAI_models_3.png" width="790" height="387"/>

## 💠 Models featured in OpenAI Research

> Read more -> [Model index for researchers](https://beta.openai.com/docs/model-index-for-researchers)

These are the most proximate models featured in our research papers that are available in the API today. Please note that not all models available in the API correspond to a paper, and even for models that are listed below there may be subtle differences that do not allow for exact replication of the paper.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/OpenAI_models_1.png" width="785" height="518"/>
<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Language%20Models/img/OpenAI_models_2.png" width="789" height="626"/>

1. This model is deprecated and listed here for historical information only.
2. These parameters are what is indicated in the **paper**, and in some cases may differ slightly from what is in the API.
3. `code-cushman-001` is a stronger, multilingual version of the Codex 12B model in [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374).

### 📄 Papers

- [[2005.14165] Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [[2107.03374] Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [[2201.10005] Text and Code Embeddings by Contrastive Pre-Training](https://arxiv.org/abs/2201.10005)
- [[2009.01325] Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [[2203.02155] Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

# ♦️ Gopher

- 📄 **Paper:** 
- 📰 **Articles:**
  - [Language modelling at scale: Gopher, ethical considerations, and retrieval](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval) by DeepMind Research

# ♦️ Megatron

- 📄 **Paper:** 
- 📰 **Articles:**
  - [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, the World’s Largest and Most Powerful Generative Language Model](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) by NVIDIA Developer

# ♦️ Pathways Language Model (PaLM)

- 📄 **Paper:** [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- 📰 **Articles:**
  - [Pathways Language Model (PaLM): Scaling to 540 Billion Parameters for Breakthrough Performance](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) by Google Research

# ♦️ Chinchilla

- 📄 **Paper:** [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556v1)

## :octocat: GitHub Repositiories

| Title | Description, Information |
| :---:         |          :--- |
|[PaLM + RLHF - Pytorch (wip)](https://github.com/lucidrains/PaLM-rlhf-pytorch)|Implementation of RLHF (Reinforcement Learning with Human Feedback) on top of the PaLM architecture. Basically ChatGPT but with PaLM|


# :octocat: GitHub Repositiories

| Title | Description, Information |
| :---:         |          :--- |
|[Awesome Transformer & Transfer Learning in NLP](https://github.com/cedrickchee/awesome-transformer-nlp)|A curated list of NLP resources focused on Transformer networks, attention mechanism, GPT, BERT, ChatGPT, LLMs, and transfer learning.|
