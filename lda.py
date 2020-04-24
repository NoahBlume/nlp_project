import nltk; nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint
import json

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

# spacy for lemmatization
import spacy

# Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

from tqdm import tqdm

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def make_bigrams(texts, data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    return [bigram_mod[doc] for doc in texts]


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, typ='original'):
    """
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    """
    coherence_values = []
    model_list = []
    topics_lis = []
    for num_topics in tqdm(range(start, limit, step)):

        if typ == 'mallet':
            # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
            mallet_path = 'mallet-2.0.8/bin/mallet'
            model = gensim.models.wrappers.LdaMallet(mallet_path, 
                                                    corpus=corpus, 
                                                    num_topics=num_topics,
                                                    iterations = 500,
                                                    id2word=dictionary)
        if typ == 'original':
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           iterations = 500,
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        topics_lis.append(coherencemodel.get_coherence_per_topic())

    return model_list, coherence_values, topics_lis

def format_topics_sentences(ldamodel, corpus, texts, typ='original'):
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    if typ == 'mallet':
        dist = ldamodel[corpus]
    if typ == 'original':
        dist = [ldamodel.get_document_topics(item) for item in corpus]
    for i, row in enumerate(dist):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                # topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4)]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Topic', 'Perc_Contribution']

    # # Add original text to the end of the output
    # contents = pd.Series(texts)
    # sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return sent_topics_df


def optimal_coherence(data, typ='original', start=2, limit=4, step=2):

    # tokenize
    data_words = list(sent_to_words(data))
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, data_words)
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'ADP'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # # Create General LDA model
    # lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
    #                                        id2word=id2word,
    #                                        num_topics=8, 
    #                                        random_state=100,
    #                                        update_every=1,
    #                                        chunksize=100,
    #                                        passes=10,
    #                                        alpha='auto',
    #                                        iterations = 500,
    #                                        per_word_topics=True)

    # # doc_lda = lda_model[corpus]
    # coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    # coherence_lda = coherence_model_lda.get_coherence()
    # print('\nCoherence Score of LDA: ', coherence_lda)

    # # Test ldamallet
    # # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    # mallet_path = 'mallet-2.0.8/bin/mallet' # update this path
    # ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, 
    #                                          corpus=corpus, 
    #                                          num_topics=20, 
    #                                          id2word=id2word,
    #                                          iterations=500)

    # coherence_model_ldamallet = CoherenceModel(model=ldamallet, 
    #                                         texts=data_lemmatized, 
    #                                         dictionary=id2word, 
    #                                         coherence='c_v')
    # coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    # print('\nCoherence Score of LDA Mallet: ', coherence_ldamallet)

    model_list, coherence_values, topic_lis = compute_coherence_values(dictionary=id2word, 
                                                            corpus=corpus, 
                                                            texts=data_lemmatized, 
                                                            limit=limit, 
                                                            start=start,
                                                            step=step,
                                                            typ=typ)

    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')

    plt.savefig("coherence_values_mallet.png")

    optimal_idx = coherence_values.index(max(coherence_values))
    optimal_model = model_list[optimal_idx]
    optimal_score = coherence_values[optimal_idx]
    optimal_topic = topic_lis[optimal_idx]

    # model_topics = optimal_model.show_topics(formatted=False)
    # pprint(optimal_model.print_topics(num_words=10))

    return corpus, optimal_model, optimal_score, optimal_idx, optimal_topic



if __name__ == "__main__":

    # Import Dataset
    df_fox = pd.read_json('filtered_fox_headlines.json')
    df_fox['source'] = 'fox'
    df_msnbc = pd.read_json('filtered_msnbc_headlines.json')
    df_msnbc['source'] = 'msnbc'
    df = pd.concat([df_msnbc, df_fox])

    data = df.headline.values.tolist()
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]

    start=11
    limit=12
    step=1

    corpus, optimal_model, optimal_score, index, optimal_topic = optimal_coherence(data, typ='mallet', start=start, limit=limit, step=step)

    model_topics = optimal_model.show_topics(formatted=False)
    dic_topics = {}
    for topic in model_topics:
        tmp = {}
        for t in topic[1]:
            tmp[str(t[0])] = str(t[1])
        dic_topics[str(topic[0])] = tmp
    with open(f'ldamallet_model_topic_{start + step * index}_groups_score_{optimal_score}.json', 
              'w') as json_file:
        json.dump(dic_topics, json_file)
    
    temp_file = datapath("model")
    optimal_model.save(temp_file)

    lda = gensim.models.ldamodel.LdaModel.load(temp_file)
    print(optimal_topic)

    sent_topics_df = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data, typ='mallet')
    df['topic'] = sent_topics_df['Topic'].astype(int)
    df['topic_contribution'] = sent_topics_df['Perc_Contribution']
    df.reset_index(inplace=True)

    df.to_json("headlines_lda_mallet.json")

