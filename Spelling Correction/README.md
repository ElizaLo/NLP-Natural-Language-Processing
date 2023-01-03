# Spelling Correction

- [Papers with Code - Spelling Correction](https://paperswithcode.com/task/spelling-correction)

## 🔹 Pure Python Spell Checking

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
- **Dataframe** 

```python
from spellchecker import SpellChecker

spell = SpellChecker()

for i in range(len(df.index)):
    df.at[i, 'SpellChecker'] = spell.correction(df['Text'][i])
```

## :octocat: Frameworks, libraries, etc.

⚙️ **Github Topics**

- [#spelling-correction](https://github.com/topics/spelling-correction)
- [#spellchecker](https://github.com/topics/spellchecker)
- [#spellcheck](https://github.com/topics/spellcheck)
- [#spell-checker](https://github.com/topics/spell-checker)


| Title | Description, Information |
| :---:         |          :--- |
|**pyspellchecker**|<p>Pure Python Spell Checking based on [Peter Norvig’s blog post](https://norvig.com/spell-correct.html) on setting up a simple spell checking algorithm.</p><p>It uses a **Levenshtein Distance** algorithm to find permutations within an edit distance of 2 from the original word. It then compares all permutations (insertions, deletions, replacements, and transpositions) to known words in a word frequency list. Those words that are found more often in the frequency list are more likely the correct results.</p><p>`pyspellchecker` supports multiple languages including English, Spanish, German, French, and Portuguese.</p><ul><li>[pyspellchecker](https://pypi.org/project/pyspellchecker/) on pypi.org</li><li>[Documentation](https://pyspellchecker.readthedocs.io/en/latest/index.html#pyspellchecker)</li><li> :octocat: [Pure Python Spell Checking](https://github.com/barrust/pyspellchecker)</li></ul>|
|**spellCheck**|<p>This package currently focuses on Out of Vocabulary (OOV) word or non-word error (NWE) correction using BERT model. The idea of using BERT was to use the context when correcting OOV. To improve this package, I would like to extend the functionality to identify RWE, optimising the package, and improving the documentation.</p><ul><li> :octocat: [spellCheck](https://github.com/R1j1t/contextualSpellCheck)</li></ul>|
|**NeuSpell: A Neural Spelling Correction Toolkit**|<p>Added support for different transformer-based models such DistilBERT, XLM-RoBERTa, etc. See Finetuning on custom data and creating new models section for more details.</p><p>Neuspell's BERT pretrained model is now available as part of huggingface models as murali1996/bert-base-cased-spell-correction.</p><ul><li> :octocat: [NeuSpell](https://github.com/neuspell/neuspell#Installation-through-pip)</li></ul>|
|**JamSpell**|<p>Modern spell checking library - accurate, fast, multi-language</p><ul><li>[jamspell.com](https://jamspell.com)</li><li>[jamspell](https://pypi.org/project/jamspell/) on pypi.org</li><li> :octocat: [JamSpell](https://github.com/bakwc/JamSpell)</li><li>[]()</li></ul>|
|**symspellpy**|<p>symspellpy is a Python port of SymSpell v6.7.1, which provides much higher speed and lower memory consumption. Unit tests from the original project are implemented to ensure the accuracy of the port.</p><p>Python port of SymSpell: 1 million times faster spelling correction & fuzzy search through Symmetric Delete spelling correction algorithm</p><ul><li>[symspellpy](https://pypi.org/project/symspellpy/) on pypi.org</li><li> :octocat; [symspellpy](https://github.com/mammothb/symspellpy)</li></ul>|
|**Manual Spell Checker**|<p>A manual spell checker built on pyenchant that allows you to swiftly correct misspelled words</p><ul><li>[manual-spellchecker 1.2](https://pypi.org/project/manual-spellchecker/) on pypi.org</li><li> :octocat: [Manual Spell Checker](https://github.com/atif-hassan/manual_spellchecker)</li></ul>|

## 📰 Articles

- [Spelling Correction Of The Text Data In Natural Language Processing](https://medium.com/@nutanbhogendrasharma/spelling-correction-of-the-text-data-in-natural-language-processing-e9848407cf3b)
- [Spelling checker in Python](https://www.geeksforgeeks.org/spelling-checker-in-python/)
