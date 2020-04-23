RELEVANT_WORDS = {'coronavirus', 'virus', 'covid', 'covid19', 'covid-19'}


def filter_articles(articles):
    filtered_articles = []

    for article in articles:
        words = article['words']
        has_relevant_word = False

        for word in words:
            if word.lower() in RELEVANT_WORDS:
                has_relevant_word = True
                break

        if has_relevant_word:
            filtered_articles.append(article)

    return filtered_articles


if __name__ == "__main__":
    pass

