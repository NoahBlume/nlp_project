import json


def fix_format(data):
    new_articles = dict()
    for i in range(0, 20):
        new_articles[i] = []

    for i in range(0, 3241):
        index = str(i)
        headline = data['headline'][index]
        source = data['source'][index]
        topic = data['topic'][index]

        article = dict()
        article['headline'] = headline
        article['source'] = source
        article['topic'] = topic
        article['sentiment_polarity'] = data['sentiment_polarity'][index]
        article['sentiment_subjectivity'] = data['sentiment_subjectivity'][index]
        article['lemmas'] = data['lemmas'][index]

        article_bucket = new_articles.get(topic, [])
        article_bucket.append(article)
        new_articles[topic] = article_bucket

    save_data("grouped_articles", new_articles)
    return new_articles


# Returns a python list or dictionary that is loaded from a local json file
#   source: (string) the news source for which articles will be loaded
def load_data(source):
    with open('./' + source + '.json') as json_file:
        data = json.load(json_file)
        # print("locally read " + source + " data:", data)
        return data


# Saves a python list or dictionary to a json file
#   source: (string) the news source that the json file will be titled after
#   data: (list or dictionary) the data to be saved to a json file
def save_data(source, data):
    # print("saving " + source + " data:", data)
    json_object = json.dumps(data, indent=4)
    with open(source + ".json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    original_lda_data = load_data("headlines_lda_original")
    new_format = fix_format(original_lda_data)
