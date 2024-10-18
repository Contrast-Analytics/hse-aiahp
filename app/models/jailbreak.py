import re
from cleantext import clean
from better_profanity import profanity
import requests
import spacy
from spacy.tokens import Token
from profanity_filter import ProfanityFilter

class Jailbreak:
    def __init__(self):
        self.url = 'https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/ru'
        self.bad_words = self.load_bad_words()
        self.nlp = spacy.load("ru_core_news_sm")
        Token.set_extension("is_profane", default=False, force=True)
        self.pf = ProfanityFilter(languages=['ru'], nlps={'ru': self.nlp})

    def load_bad_words(self):
        response = requests.get(self.url)
        return response.text.splitlines()

    def remove_code_from_text(self, text):
        code_pattern = re.compile(r'\b(def|class|if|else|elif|for|while|try|except|finally|with|import|from|print|return|break|continue|pass|raise|assert|yield|async|await|lambda|global|nonlocal|del|exec|eval)\b.*', re.DOTALL)
        fenced_code_pattern = re.compile(r'```.*?```', re.DOTALL)
        cleaned_text = code_pattern.sub('', text)
        cleaned_text = fenced_code_pattern.sub('', cleaned_text)
        return cleaned_text

    def remove_obscene_lexicon(self, text):
        cleaned_text = clean(text, extra_spaces=True, stemming=False, stopwords=True, lowercase=False, stp_lang='english')
        cleaned_text = profanity.censor(cleaned_text)
        cleaned_text = self.pf.censor(cleaned_text)
        bad_words_re = re.compile(r'\b(' + '|'.join(re.escape(word) for word in self.bad_words) + r')\b', re.IGNORECASE)
        cleaned_text = bad_words_re.sub(lambda match: '*' * len(match.group()), cleaned_text)
        return cleaned_text

    def clean_answer(self, text):
        text = self.remove_code_from_text(text)
        text = self.remove_obscene_lexicon(text)
        return text

# Пример использования
if __name__ == "__main__":
    jailbreak = Jailbreak()
    sample_text = """
    Этот текст содержит ненормативную лексику и код:
    ```python
    def bad_function():
        print("Это плохой код")
    ```
    А также ненормативную лексику: блядь, ебать.
    """
    cleaned_text = jailbreak.clean_answer(sample_text)
    print(cleaned_text)