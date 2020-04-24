from headline_retriever import load_articles, save_articles
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date, datetime

CUTOFF_DATE = date(2020, 2, 28)  # last date to include articles from (inclusive) - first COVID-19 death was 2/29


def do_nothing(already_tokenized_headline):
    return already_tokenized_headline


VECTORIZER = TfidfVectorizer(analyzer='word',
                             tokenizer=do_nothing,
                             preprocessor=do_nothing,
                             token_pattern=None)


def fit_vectorizer(article_lists):
    combined_headlines = []
    for articles in article_lists:
        for article in articles:
            combined_headlines.append(article['lemmas'])

    # tokenize and build vocab
    VECTORIZER.fit(combined_headlines)


def add_tfidf_vectors(articles):
    for article in articles:
        headline = article['lemmas']
        headline_vector = list(VECTORIZER.transform([headline]).toarray()[0])
        article['tfidf_vector'] = headline_vector


def apply_cutoff(articles, cutoff_date=CUTOFF_DATE):
    if cutoff_date is None:
        return articles

    cutoff_articles = []
    for article in articles:
        article_date = datetime.strptime(article['date'], '%Y-%m-%d').date()
        if article_date <= cutoff_date:
            cutoff_articles.append(article)
    return cutoff_articles


if __name__ == "__main__":
    pass

    # example commands:
#     fox_articles = load_articles("foxnews_pre_processed")
#     msnbc_articles = load_articles("msnbc_pre_processed")

#     fit_vectorizer([fox_articles, msnbc_articles])

#     add_tfidf_vectors(fox_articles)
#     add_tfidf_vectors(msnbc_articles)

#     save_articles("foxnews_tfidf_added", fox_articles)
#     save_articles("msnbc_tfidf_added", msnbc_articles)
