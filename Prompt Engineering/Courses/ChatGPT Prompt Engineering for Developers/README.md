# ChatGPT Prompt Engineering for Developers

> by DeepLearning.AI and OpenAI

- [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)

## 🗒️ Notes

### 🔹 Prompts

**Review:**

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
**Prompt:**
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
**Prompt:**
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
**Prompt:**
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

**Review:**

```python

review_1 = lamp_review 

# review for a standing lamp
review_2 = """
Needed a nice lamp for my bedroom, and this one \
had additional storage and not too high of a price \
point. Got it fast - arrived in 2 days. The string \
to the lamp broke during the transit and the company \
happily sent over a new one. Came within a few days \
as well. It was easy to put together. Then I had a \
missing part, so I contacted their support and they \
very quickly got me the missing piece! Seems to me \
to be a great company that cares about their customers \
and products. 
"""

# review for an electric toothbrush
review_3 = """
My dental hygienist recommended an electric toothbrush, \
which is why I got this. The battery life seems to be \
pretty impressive so far. After initial charging and \
leaving the charger plugged in for the first week to \
condition the battery, I've unplugged the charger and \
been using it for twice daily brushing for the last \
3 weeks all on the same charge. But the toothbrush head \
is too small. I’ve seen baby toothbrushes bigger than \
this one. I wish the head was bigger with different \
length bristles to get between teeth better because \
this one doesn’t.  Overall if you can get this one \
around the $50 mark, it's a good deal. The manufactuer's \
replacements heads are pretty expensive, but you can \
get generic ones that're more reasonably priced. This \
toothbrush makes me feel like I've been to the dentist \
every day. My teeth feel sparkly clean! 
"""

# review for a blender
review_4 = """
So, they still had the 17 piece system on seasonal \
sale for around $49 in the month of November, about \
half off, but for some reason (call it price gouging) \
around the second week of December the prices all went \
up to about anywhere from between $70-$89 for the same \
system. And the 11 piece system went up around $10 or \
so in price also from the earlier sale price of $29. \
So it looks okay, but if you look at the base, the part \
where the blade locks into place doesn’t look as good \
as in previous editions from a few years ago, but I \
plan to be very gentle with it (example, I crush \
very hard items like beans, ice, rice, etc. in the \ 
blender first then pulverize them in the serving size \
I want in the blender then switch to the whipping \
blade for a finer flour, and use the cross cutting blade \
first when making smoothies, then use the flat blade \
if I need them finer/less pulpy). Special tip when making \
smoothies, finely cut and freeze the fruits and \
vegetables (if using spinach-lightly stew soften the \ 
spinach then freeze until ready for use-and if making \
sorbet, use a small to medium sized food processor) \ 
that you plan to use that way you can avoid adding so \
much ice if at all-when making your smoothie. \
After about a year, the motor was making a funny noise. \
I called customer service but the warranty expired \
already, so I had to buy another one. FYI: The overall \
quality has gone done in these types of products, so \
they are kind of counting on brand recognition and \
consumer loyalty to maintain sales. Got it in about \
two days.
"""

reviews = [review_1, review_2, review_3, review_4]
```

**Prompt:**

```python
for i in range(len(reviews)):
    prompt = f"""
    What the writer is expressing?\
    The review is delimited with triple backticks. \
    Give your answer as with yes or no and explain why by \
    citing the writer.
    The answer should consist of all positive and negative \ 
    experiences of the writer: what the writer says about quality, \
    delivery, company, etc.? Name to what it belongs by yourself. \

    Structure the answer in the format:

    To what it belongs: buttery
    Citing: "cite", "cite", etc.

    To what it belongs: quality
    Citing: "cite", "cite", etc.

    To what it belongs: shape
    Citing: "cite", "cite", etc.

    To what it belongs: 
    Citing: "cite", "cite", etc. 

    Review: ```{reviews[i]}```
    """

    response = get_completion(prompt)
    print(i, response, "\n")


response = get_completion(prompt)
print(response)
```

**Output:**
```
0 To what it belongs: delivery
Citing: "Got it fast."

To what it belongs: quality
Citing: "The string to our lamp broke during the transit", "It was easy to put together."

To what it belongs: company
Citing: "the company happily sent over a new one", "Lumina seems to me to be a great company that cares about their customers and products!!"

To what it belongs: N/A
Citing: "Needed a nice lamp for my bedroom, and this one had additional storage and not too high of a price point.", "I had a missing part, so I contacted their support and they very quickly got me the missing piece!" 

1 To what it belongs: delivery
Citing: "Got it fast - arrived in 2 days."

To what it belongs: quality
Citing: "The string to the lamp broke during the transit and the company happily sent over a new one."

To what it belongs: company
Citing: "Seems to me to be a great company that cares about their customers and products." 

To what it belongs: N/A
Citing: "Needed a nice lamp for my bedroom, and this one had additional storage and not too high of a price point. Came within a few days as well. It was easy to put together. Then I had a missing part, so I contacted their support and they very quickly got me the missing piece!" 

2 To what it belongs: quality
Citing: "The battery life seems to be pretty impressive so far."

To what it belongs: shape
Citing: "But the toothbrush head is too small. I’ve seen baby toothbrushes bigger than this one. I wish the head was bigger with different length bristles to get between teeth better because this one doesn’t."

To what it belongs: company
Citing: "The manufactuer's replacements heads are pretty expensive, but you can get generic ones that're more reasonably priced."

To what it belongs: positive experience
Citing: "This toothbrush makes me feel like I've been to the dentist every day. My teeth feel sparkly clean!"

To what it belongs: recommendation
Citing: "Overall if you can get this one around the $50 mark, it's a good deal." 

3 To what it belongs: delivery
Citing: "Got it in about two days."

To what it belongs: quality
Citing: "the part where the blade locks into place doesn’t look as good as in previous editions from a few years ago", "The overall quality has gone done in these types of products"

To what it belongs: shape
Citing: "the part where the blade locks into place doesn’t look as good as in previous editions from a few years ago"

To what it belongs: company
Citing: "The overall quality has gone done in these types of products, so they are kind of counting on brand recognition and consumer loyalty to maintain sales." 

To what it belongs: delivery
Citing: "Got it in about two days."

To what it belongs: quality
Citing: "the part where the blade locks into place doesn’t look as good as in previous editions from a few years ago", "The overall quality has gone done in these types of products"

To what it belongs: shape
Citing: "the part where the blade locks into place doesn’t look as good as in previous editions from a few years ago"

To what it belongs: company
Citing: "The overall quality has gone done in these types of products, so they are kind of counting on brand recognition and consumer loyalty to maintain sales."
```
---
