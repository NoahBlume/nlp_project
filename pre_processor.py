import stanza
from headline_retriever import load_articles, collect_articles, save_articles
from textblob import TextBlob
from datetime import date

NLP = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner')

END_DATE = date(2020, 3, 27)  # the chosen last day to retrieve article headlines


# in place modification of a list of article dictionaries
def pre_process(articles):
    for article in articles:
        headline = article['headline']

        sentiment = TextBlob(headline).sentiment
        # print("sentiment:", sentiment)
        sentiment_polarity = sentiment[0]  # range from -1 to 1. -1 being the most negative, 1 being the most positive
        sentiment_subjectivity = sentiment[1]  # range from 0 to 1. 0 being factual, 1 being an opinion

        processed_headline = NLP(headline)

        words = []
        lemmas = []
        pos = []
        entities = processed_headline.entities

        entity_dicts = []
        for entity in entities:
            entity_dict = dict()

            entity_dict['text'] = entity.text
            entity_dict['type'] = entity.type
            entity_dict['start_char'] = entity.start_char
            entity_dict['end_char'] = entity.end_char

            entity_dicts.append(entity_dict)

        for sentence in processed_headline.sentences:
            for word in sentence.words:
                words.append(word.text)
                lemmas.append(word.lemma)
                pos.append(word.pos)

        article['sentiment_polarity'] = sentiment_polarity
        article['sentiment_subjectivity'] = sentiment_subjectivity
        article['words'] = words
        article['lemmas'] = lemmas
        article['pos'] = pos
        article['entities'] = entity_dicts


def average_sentiments(preprocessed_articles):
    if len(preprocessed_articles) < 1:
        print("avg polarity:", 0)
        print("avg subjectivity:", 0)
        return

    total_subjectivity = 0
    total_polarity = 0
    for article in preprocessed_articles:
        total_polarity += article['sentiment_polarity']
        total_subjectivity += article['sentiment_subjectivity']

    print("avg polarity:", total_polarity/len(preprocessed_articles))
    print("avg subjectivity:", total_subjectivity/len(preprocessed_articles))


def average_words_per_headline(preprocessed_articles):
    total_words = 0
    for article in preprocessed_articles:
        total_words += len(article['headline'].split())

    print("avg words:", total_words/len(preprocessed_articles))


if __name__ == "__main__":
    pass
    #Example commands contained below:
    
    # attempt to load in the article data if it exists
#     fox_articles = load_articles("foxnews_headlines")
#     msnbc_articles = load_articles("msnbc_headlines")

#     pre_process(fox_articles)
#     pre_process(msnbc_articles)

#     save the retrieved article data
#     save_articles("foxnews_pre_processed", fox_articles)
#     save_articles("msnbc_pre_processed", msnbc_articles)
