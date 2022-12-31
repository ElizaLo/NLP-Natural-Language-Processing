# Text Pre-Processing

- Lowercase
- Remove `stopwords`
- Punctuations
- Remove Numbers
- Information Extraction _(remove phone numbers, emails, etc.)_
- Remove HTML Code and URL Links
- Spell Checks
- Remove Emoji 😄
- Tokeniation
- Normalization
    - Stemming
    - Lemmatization
- Synonyms, Antonyms, Hypernyms
- Part of Speech Tagging (POS)   

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

 As `string.punctuation` in Python contains these symbols - `!"#$%&\'()*+,-./:;?@[\\]^_{|}~`.

## Remove Numbers

## 🔹 Stopwords

English is one of the most common languages, especially in the world of social media. For instance, _"a," "our," "for," "in," etc._ are in the set of most commonly used words. Removing these words helps the model to consider only key features. These words also don't carry much information. By eliminating them, data scientists can focus on the important words.

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

## 🔹 Remove Emoji 😄

- [Remove emoji from a text file](https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b)

## 🔹 Spell Checks

### 💠 Pure Python Spell Checking

```python
from spellchecker import SpellChecker

spelling = SpellChecker()

def spelling_checks(text):
    correct_result = []
    typo_words = spelling.unknown(text.split())
    for word in text.split():
        if word in typo_words:
            correct_result.append(spelling.correction(word))
        else:
            correct_result.append(word)
    return " ".join(correct_result)
        
text = input('Your-Text: ')
print('Error free text:',spelling_checks(text))
```

### :octocat: Frameworks, libraries, etc.

| Title | Description, Information |
| :---:         |          :--- |
|**pyspellchecker**|<p>Pure Python Spell Checking based on [Peter Norvig’s blog post](https://norvig.com/spell-correct.html) on setting up a simple spell checking algorithm.</p><p>It uses a **Levenshtein Distance** algorithm to find permutations within an edit distance of 2 from the original word. It then compares all permutations (insertions, deletions, replacements, and transpositions) to known words in a word frequency list. Those words that are found more often in the frequency list are more likely the correct results.</p><p>`pyspellchecker` supports multiple languages including English, Spanish, German, French, and Portuguese.</p><ul><li>[pyspellchecker](https://pypi.org/project/pyspellchecker/) on pypi.org</li><li>[Documentation](https://pyspellchecker.readthedocs.io/en/latest/index.html#pyspellchecker)</li><li> :octocat: [Pure Python Spell Checking](https://github.com/barrust/pyspellchecker)</li></ul>|

## 🔹 Tokeniation

Tokenizing is like splitting a whole sentence into words. You can consider a simple separator for this purpose. But a separator will fail to split the abbreviations separated by "." or special characters, like U.A.R.T., for example. Challenges increase when more languages are included. 

Most of these problems can be solved by using the `nltk` library. The `word_tokenize` module breaks the words into tokens and these words act as an input for the normalization and cleaning process. It can further be used to convert a string (text) into numeric data so that machine learning models can digest it.

## 🔹 Normalization

Normalization is an advanced step in cleaning to maintain uniformity. It brings all the words under on the roof by adding **stemming** and **lemmatization**. Many people often get stemming and lemmatizing confused. It's true that they are both normalization processes, but they are a lot different.

<img src="https://raw.githubusercontent.com/ElizaLo/NLP-Natural-Language-Processing/master/Text%20Pre-Processing/img/stemming_lemmatization.JPG" width="1011" height="326"/>

### 💠 Stemming

There are many variations of words that do not bring any new information and create redundancy, ultimately bringing ambiguity when training machine learning models for predictions. Take _"He likes to walk"_ and _"He likes walking"_, for example. Both have the same meaning, so the `stemming` function will remove the suffix and convert _"walking"_ to _"walk"_. The example in this guide uses the `PorterStemmer` module to conduct the process. You can use the `snowball` module for different languages.

### 💠 Lemmatization

Unlike stemming, lemmatization performs normalization using vocabulary and morphological analysis of words. Lemmatization aims to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma. Lemmatization uses a dictionary, which makes it slower than stemming, however the results make much more sense than what you get from stemming. Lemmatization is built on WordNet's built-in morphy function, making it an intelligent operation for text analysis. A [WordNet](https://wordnet.princeton.edu/) module is a large and public lexical database for the English language. Its aim is to maintain the structured relationship between the words. The `WordNetLemmitizer()` is the earliest and most widely used function.

```python
lemmatizer = WordNetLemmatizer()
lemmawords = [lemmatizer.lemmatize(w) for w in tokenwords]
print ('Lemmtization-form',lemmawords)
```

## 🔹 Synonyms, Antonyms, Hypernyms

The NLTK corpus reader uses a lexical database to find a word's **synonyms, antonyms, hypernyms**, etc. In this use case, you will find the synonyms _(words that have the same meaning)_ and hypernyms _(words that give a broader meaning)_ for a word by using the `synset()` function.

```python
from nltk.corpus import wordnet as wn

for ssn in wn.synsets('aid'):
    print('\nName:',ssn.name(),'\n-Abstract term: ',ssn.hypernyms(),'\n-Specific term:',ssn.hyponyms()) # Try:ssn.root_hypernyms()
```

There are three abstract terms for the word _"aid"_. The `definition()` and `examples()` functions in WordNet will help clarify the context.

```python
print('Meaning:' ,wn.synset('aid.n.01').definition()) # try any term-eg: care.n.01
print('Example: ',wn.synset('aid.n.01').examples())
```

## 🔹 Part of Speech Tagging (POS)

In the English language, one word can have different grammatical contexts, and in these cases it's not a good practice to consider the two words redundant. POS aims to make them grammatically unique.

```python
text_words = word_tokenize(text)
nltk.pos_tag(text_words)
```

## :octocat: Frameworks, libraries, etc.

| Title | Description, Information |
| :---:         |          :--- |
|**NeatText**|<p>A simple NLP package for cleaning textual data and text preprocessing. Simplifying Text Cleaning For NLP & ML</p><ul><li>[neattext](https://jcharis.github.io/neattext/) - Documentation</li><li> :octocat: [NeatText](https://github.com/Jcharis/neattext)</li></ul>|

## 📰 Articles

- [Importance of Text Pre-processing](https://www.pluralsight.com/guides/importance-of-text-pre-processing)
- [Text Cleaning Methods in NLP](https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/#h2_13)
