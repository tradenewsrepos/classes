from simplemma import text_lemmatizer
import re
import pickle

import sys

def preprocess_text(text):
    
    clean_text =re.sub('[.,:«»;%©?*,!@#$%^&()\t\n]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)
    
    # remove punctuation
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    
    # change multispaces to single space
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # delete collocations with years
    pattern = r"\b(?:(\d{2}|\d{4})\s*(?:год(?:а|у)?|year))\b"
    clean_text = re.sub(pattern, "", clean_text)
    
    # delete collocations with separate years
    pattern = r"\b(?:(\d{2}|\d{4}))\b"
    clean_text = re.sub(pattern, "", clean_text)
    
    # delete months
    pattern = r'''January|February|March|April|May|June|July|August|September|October|November|December|январ[ья]|феврал[ья]|март[а]?|апрел[ья]|мая?|июн[ья]?(?:[яю]|е[ао])?|июл[ья]?[яи]?|август[а]?|сентябр[ья]?|октябр[ья]?|ноябр[ья]?|декабр[ья]'''
    clean_text = re.sub(pattern, "", clean_text, flags = re.I)
    
    return clean_text    

class LemmaTokenizer():
    def __init__(self):
        self.lem = text_lemmatizer
        self.preprocess = preprocess_text
    
    def __call__(self, doc):
        return self.lem(self.preprocess(doc), lang = ('ru', 'en'))

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "tfidf_run"
        return super().find_class(module, name)    

class TfidfInferer:
    def __init__(self, path_to_model: str):
        # torch.cuda.is_available()
        # device=torch.cuda.current_device()
        path_to_model +='/tfidf_logreg_news_classifier_030723.pkl'
        with open(path_to_model, 'rb') as f: 
            unpickler = MyCustomUnpickler(f)
            self.pipe = unpickler.load() 
        
    def infer(self, text: str):
        res = self.pipe.predict_proba([text])[0]
        return {"score 1": res[0], "score 2": res[1]} 