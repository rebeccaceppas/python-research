text = 'This is my test text. We\'re keeping this text short to keep things manageable.'

def count_words(text):
    '''Count number of times a word shows up in a string.
     Return dictionary where keys are unique words and values are word counts.
     Skips punctuations and doesn't care about capitalization.'''

    text = text.lower()
    skip = ['.',',',':',';','"','\'']
    for punc in skip:
        text = text.replace(punc, '')
    word_counts = {}
    for word in text.split(' '):
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    return word_counts

count_words(text)

from collections import Counter

def count_words_fast(text):
    '''Count number of times a word shows up in a string using counter module.
     Return dictionary where keys are unique words and values are word counts.
     Skips punctuations and doesn't care about capitalization.'''

    text = text.lower()
    skip = ['.',',',':',';','"','\'']
    for punc in skip:
        text = text.replace(punc, '')
    word_counts = Counter(text.split(' '))

    return word_counts

def read_book(title_path):
    ''' Read book and return it as a string'''

    with open(title_path, 'r', encoding='utf8') as current_file:
        text = current_file.read()
        text = text.replace('\n','').replace('\r','')

    return text

def word_stats(word_counts):
    ''' Return number of unique words and word frequencies for each of those'''

    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)

text = read_book('./Books/English/shakespeare/Romeo and Juliet.txt')
word_counts = count_words(text)
(num_unique, counts) = word_stats(word_counts)

text = read_book('./Books/German/shakespeare/Romeo und Julia.txt')
word_counts = count_words(text)
(num_unique, counts) = word_stats(word_counts)

import os
import pandas as pd
book_dir = './Books'
stats = pd.DataFrame(columns=('language','author','title','length','unique'))
title_num = 1

for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + '/' + language):
        for title in os.listdir(book_dir + '/' + language + '/' + author):
            inputfile = book_dir + '/' + language + '/' + author + '/' + title
            text = read_book(inputfile)
            (num_unique, counts) = word_stats(count_words(text))
            stats.loc[title_num] = language, author.capitalize(), title.replace('.txt',''), sum(counts), num_unique
            title_num += 1

from matplotlib import pyplot

pyplot.plot(stats.length, stats.unique, 'bo')
pyplot.savefig('length vs unique.jpg')
pyplot.show()

pyplot.figure(figsize=(10,10))
subset = stats[stats.language=='English']
pyplot.loglog(subset.length, subset.unique, 'o', label='English', color='crimson')
subset = stats[stats.language=='French']
pyplot.loglog(subset.length, subset.unique, 'o', label='French', color='forestgreen')
subset = stats[stats.language=='German']
pyplot.loglog(subset.length, subset.unique, 'o', label='German', color='orange')
subset = stats[stats.language=='Portuguese']
pyplot.loglog(subset.length, subset.unique, 'o', label='Portuguese', color='blueviolet')
pyplot.legend()
pyplot.xlabel('Book length')
pyplot.ylabel('Number of unique words')
pyplot.savefig('Language plot.jpg')
pyplot.show()