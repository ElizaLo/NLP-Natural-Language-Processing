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

- 

| Title | Description, Information |
| :---:         |          :--- |
|**pyspellchecker**|<p>Pure Python Spell Checking based on [Peter Norvig’s blog post](https://norvig.com/spell-correct.html) on setting up a simple spell checking algorithm.</p><p>It uses a **Levenshtein Distance** algorithm to find permutations within an edit distance of 2 from the original word. It then compares all permutations (insertions, deletions, replacements, and transpositions) to known words in a word frequency list. Those words that are found more often in the frequency list are more likely the correct results.</p><p>`pyspellchecker` supports multiple languages including English, Spanish, German, French, and Portuguese.</p><ul><li>[pyspellchecker](https://pypi.org/project/pyspellchecker/) on pypi.org</li><li>[Documentation](https://pyspellchecker.readthedocs.io/en/latest/index.html#pyspellchecker)</li><li> :octocat: [Pure Python Spell Checking](https://github.com/barrust/pyspellchecker)</li></ul>|
|**Manual Spell Checker**|<p>A manual spell checker built on pyenchant that allows you to swiftly correct misspelled words</p><ul><li>[manual-spellchecker 1.2](https://pypi.org/project/manual-spellchecker/) on pypi.org</li><li> :octocat: [Manual Spell Checker](https://github.com/atif-hassan/manual_spellchecker)</li></ul>|

## 📰 Articles

- []()
