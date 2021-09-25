from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import udhr
from collections import Counter

def get_freqs(corpus, puncts):
    freqs = {}
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    a = corpus.lower()
    for number in numbers:
        a = a.replace(number,' ')
    for punctuation in puncts:
        a = a.replace(punctuation, ' ')
    a = a.replace('\n', ' ')
    corpus2 = a.split(' ')
    for item in corpus2:
        if item in freqs:
            freqs[item]+= 1
        else:
            freqs[item] = 1
    freqs.pop('')
    ### BEGIN SOLUTION
    ### END SOLUTION
    return freqs

def get_top_10(freqs):
    top_10 = []
    ### BEGIN SOLUTION
    count = Counter(freqs)
    high = count.most_common(10)
    for element in high:
        top_10.append(element[0])

    ### END SOLUTION
    return top_10

def get_bottom_10(freqs):
    bottom_10 = []
    ### BEGIN SOLUTION
    count = Counter(freqs)
    low = count.most_common()[:-11:-1]
    for element in low:
        bottom_10.append(element[0])
    ### END SOLUTION
    return bottom_10

def get_percentage_singletons(freqs):
    ### BEGIN SOLUTION
    count = Counter(freqs.values())
    length = len(freqs.values())
    return float(count[1]/length) * 100
    ### END SOLUTION
    pass

def get_freqs_stemming(corpus, puncts):
    ### BEGIN SOLUTION
    freqs = {}
    porter = PorterStemmer()
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    a = corpus.lower()
    for number in numbers:
        a = a.replace(number,' ')
    for punctuation in puncts:
        a = a.replace(punctuation, ' ')
    a = a.replace('\n', ' ')
    corpus2 = a.split(' ')
    for item in corpus2:
        item = porter.stem(item)
        if item in freqs:
            freqs[item]+= 1
        else:
            freqs[item] = 1
    freqs.pop('')
    ### END SOLUTION
    return freqs

def get_freqs_lemmatized(corpus, puncts):
    ### BEGIN SOLUTION
    wordnet_lemmatizer = WordNetLemmatizer()
    freqs = {}
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    a = corpus.lower()
    for number in numbers:
        a = a.replace(number,' ')
    for punctuation in puncts:
        a = a.replace(punctuation, ' ')
    a = a.replace('\n', ' ')
    corpus2 = a.split(' ')
    for item in corpus2:
        item = wordnet_lemmatizer.lemmatize(item, pos="v")
        if item in freqs:
            freqs[item]+= 1
        else:
            freqs[item] = 1
    freqs.pop('')
    ### END SOLUTION
    return freqs

def size_of_raw_corpus(freqs):
    ### BEGIN SOLUTION

    ### END SOLUTION
    return len(freqs.keys())

def size_of_stemmed_raw_corpus(freqs_stemming):
    ### BEGIN SOLUTION
    ### END SOLUTION
    return len(freqs_stemming.keys())

def size_of_lemmatized_raw_corpus(freqs_lemmatized):
    ### BEGIN SOLUTION
    ### END SOLUTION
    return len(freqs_lemmatized.keys())

def percentage_of_unseen_vocab(a, b, length_i):
    ### BEGIN SOLUTION
    return float(len(set(a) - set(b))/length_i)


    ### END SOLUTION
    pass

def frac_80_perc(freqs):
    ### BEGIN SOLUTION
    sum1 = sum(freqs.values())
    eighty_percent = 0.8 * sum1
    count1 = 0
    count = Counter(freqs)
    a = count.most_common()
    i = 0
    while count1 < eighty_percent:
        count1 = count1 + a[i][1]
        i += 1
    ### END SOLUTION
    return float(i / len(freqs.keys()))
    pass

def plot_zipf(freqs):
    ### BEGIN SOLUTION
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.

def get_TTRs(languages):
    TTRs = {}
    for lang in languages:
        words = udhr.words(lang)
        ### BEGIN SOLUTION
        set1 = set()
        arr = []
        x = 0
        for i in range(len(words)):
            set1.add(words[i].lower())
            if x == 13:
                break
            if i == 99 + (100 * x):
                x += 1
                arr.append(len(set1))
        TTRs[lang] = arr
        ### END SOLUTION
    return TTRs

def plot_TTRs(TTRs):
    ### BEGIN SOLUTION
    ### END SOLUTION
    plt.show()  # put this line at the end to display the figure.
