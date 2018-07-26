import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def cleantext(string):
    """
    remove symbol, number etc
    """
    string = re.sub(r"(\s)@\w+","",string)
    string = re.sub(r"[^a-z']", " ", string)
    string = re.sub(r'\.+', ". ", string)
    string = re.sub(r'\!+', "! ", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"can\'t", " can not", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", ",", string)
    string = re.sub(r"!", ".", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def preprocessing(doc):
    """
    input: array of string
    output: array of array of string
    """
    doc = doc.lower()
    doc = cleantext(doc)
    doc = [wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(doc)]
    return [w for w in doc if w not in stop_words]



def cutoff_word_frequency(corpus, lb=0.1, ub = 0.6):
    """
    remove word occur in less than lb(lb < 0) of documents or more than ub(ub < 0) of documents
    """
    total = len(corpus)
    cnt = Counter(sum([list(set(l)) for l in corpus], []))
    filtered_word = set([word for word, c in cnt.items() if c/total < lb or c/total > ub])
    return [[w for w in l if w not in filtered_word] for l in corpus]
