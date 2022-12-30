# Text Pre-Processing

- Lowercase
- Remove `stopwords`
- Punctuations
- Information Extraction _(remove phone numbers, emails, etc.)_
- Remove HTML Code and URL Links

## 🔹 Lowercase

```python
def lowercase(text):
    return text.lower() 
```

## 🔹 Punctuations

Let's check the types of punctuation the `string.punctuation()` function filters out. To achieve the punctuation removal, `maketrans()` is used. It can replace the specific characters' punctuation, in this case with some other character. The code replaces the punctuation with spaces (`''`). `translate()` is a function used to make these replacements.

```python
output = string.punctuation

print('list of punctuations:', output)

def punctuation_cleaning(intext):
    return text.translate(str.maketrans('', '', output))
    
print('\nNo-punctuation:',punctuation_cleaning(clean_text))
```

## 🔹 Stopwords

### 📰 Articles

- [Text pre-processing: Stop words removal using different libraries](https://towardsdatascience.com/text-pre-processing-stop-words-removal-using-different-libraries-f20bac19929a)

## 🔹 Information Extraction

### 📰 Articles

- [Remove personal information from a text with Python — Part II](https://towardsdatascience.com/remove-personal-information-from-a-text-with-python-part-ii-ner-2e6529d409a6) - Implementation of a privacy filter in Python that removes Personal Identifiable Information (PII) with Named Entity Recognition (NER)

## 🔹 HTML Code and URL Links

The code below uses regular expressions (`re`). To perform matches with a regular expression, use `re.complie` to convert them into objects so that searching for patterns becomes easier and string substitution can be performed. A `.sub()` function is used for this.

```python
def url_remove(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def html_remove(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

text1 = input('Your-Text:')
print('\nNo-url-links:', url_remove(text1))

text2 = input('Your Text:')
print('\nNo-html-codes:', html_remove(text2))
```

## 🔹 Spell Checks

## :octocat: Frameworks, libraries, etc.

| Title | Description, Information |
| :---:         |          :--- |
|**NeatText**|<p>A simple NLP package for cleaning textual data and text preprocessing. Simplifying Text Cleaning For NLP & ML</p><ul><li>[neattext](https://jcharis.github.io/neattext/) - Documentation</li><li> :octocat: [NeatText](https://github.com/Jcharis/neattext)</li></ul>|

## 📰 Articles

- [Importance of Text Pre-processing](https://www.pluralsight.com/guides/importance-of-text-pre-processing)
