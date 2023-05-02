# ChatGPT Prompt Engineering for Developers

> by DeepLearning.AI and OpenAI

- [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)

## 🗒️ Notes

### Prompts

```python
lamp_review = """
Needed a nice lamp for my bedroom, and this one had \
additional storage and not too high of a price point. \
Got it fast.  The string to our lamp broke during the \
transit and the company happily sent over a new one. \
Came within a few days as well. It was easy to put \
together.  I had a missing part, so I contacted their \
support and they very quickly got me the missing piece! \
Lumina seems to me to be a great company that cares \
about their customers and products!!
"""
```
---
**Input:**
```python
f"""
What the writer is expressing? The review is delimited with triple backticks. Give your answer as with yes or no and explain why by \
citing the writer.
The answer should consist of all positive and negative experiences of the writer: what the writer says about quality, \
delivery, company, etc.? 
Name to what it belongs by yourself. It could belong not only to company, delivery and quality.

Format the answer:
To what it belongs: company
Citing: "cite", "cite", etc.

To what it belongs: delivery
Citing: "cite", "cite", etc.

To what it belongs: quality
Citing: "cite", "cite", etc.

To what it belongs: 
Citing: "cite", "cite", etc.

To what it belongs: 
Citing: "cite", "cite", etc.

Review text: '''{lamp_review}'''
"""
```

**Output:**
```
To what it belongs: company
Citing: "Lumina seems to me to be a great company that cares about their customers and products!!"

To what it belongs: delivery
Citing: "Got it fast.", "Came within a few days as well."

To what it belongs: quality
Citing: "Needed a nice lamp for my bedroom", "It was easy to put together."

To what it belongs: customer service
Citing: "the company happily sent over a new one", "they very quickly got me the missing piece!"
```
---
**Input:**

```
'''
Your task is to extract relevant information \
(extract phrases, not the whole sentence) from \ 
a product review from an ecommerce site to give \
feedback to the Feedback department. Extract what exactly customers like and dislike.

From the review below, delimited by triple quotes \
extract the information relevant to shipping and \ 
delivery. Limit to 30 words.
'''
```

**Output:**
```
0 Likes: 
- "soft and super cute"
- "face has a friendly look"
- "arrived a day earlier than expected"

Dislikes:
- "a bit small for what I paid"
- "other options that are bigger for the same price" 

1 Likes: 
- Lamp has additional storage
- Reasonable price
- Fast delivery (2 days)
- Great customer service, company cares about customers and products

Dislikes: 
- String to the lamp broke during transit (but was quickly replaced)
- Missing part (but was quickly provided by customer support) 

2 Likes: 
- Battery life is impressive
- Toothbrush makes teeth feel sparkly clean

Dislikes:
- Toothbrush head is too small
- Bristles are not long enough to get between teeth well
- Manufacturer's replacement heads are expensive 

3 Likes: 
- Seasonal sale price of $49 for the 17 piece system
- Special tip for making smoothies with frozen fruits and vegetables

Dislikes: 
- Price increase from $70-$89 for the same system in December
- Base of the system doesn't look as good as previous editions
- Motor started making a funny noise after a year of use
- Overall quality of the product has gone down
```
---
**Input:**

```
"""
Is the writer of the following review expressing happiness?\
The review is delimited with triple backticks. \
Give your answer as with yes or no and explain why by \
citing the writer.

Review text: '''{lamp_review}'''
"""
```

**Output:**
```
Yes, the writer is expressing happiness. The writer mentions that they are \
happy with the lamp, the fast delivery, and the company's customer service. \
They also use exclamation marks and positive language, such as "great company" \ 
and "cares about their customers and products."
```
---
