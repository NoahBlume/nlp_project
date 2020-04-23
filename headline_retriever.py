import urllib.request, json
from datetime import date, timedelta

API_KEY = "YOUR API KEY HERE"
DEFAULT_START_DATE = date(2019, 11, 1)
ARTICLE_LIMIT = "100"  # max supported per query is 100
SEARCH_TERM = "coronavirus"
SORT_BY = "relevancy"


# Returns a dictionary of news articles retrieved from newsapi.org
# The API will only return about 100 articles at a time
#   source_domain: (string) the domain name of the news source (eg 'msnbc.com')
#   from_date: (datetime.date) the first date to retrieve news articles from
#   to_date: (datetime.date) the last date to retrieve news articles from
def get_headlines(source_domain, from_date=None, to_date=None):
    # construct the request url
    request_url = "http://newsapi.org/v2/everything?q=" + SEARCH_TERM + "&pageSize=" + ARTICLE_LIMIT \
                  + "&sortBy=" + SORT_BY + "&domains=" + source_domain
    if from_date is not None and to_date is not None:
        request_url += "&from=" + str(from_date) + "&to=" + str(to_date)
    request_url += "&apiKey=" + API_KEY

    # makes the request, converts the json to a python dictionary, and returns the dictionary
    with urllib.request.urlopen(request_url) as url:
        return json.loads(url.read().decode())


# Returns a list of dates over the given range, with a step of the given delta
#   start_date: (datetime.date) the first date to include in the list
#   end_date: (datetime.date) the final date to include in the list
#   delta: (datetime.timedelta) the number of days between each date in the list
def datespan(start_date=DEFAULT_START_DATE, end_date=date.today(), delta=timedelta(days=1)):
    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date)
        current_date += delta
    return dates


# Returns a list of articles from the given source_domain over the given date range
# Makes multiple calls to get_headlines, which has a limit to the number of articles it returns with each call
# collect_articles itself is not limited in the number of articles it can return
#   source_domain: (string) the domain name of the news source (eg 'msnbc.com')
#   start_date: (datetime.date) the first date to collect articles from
#   end_date: (datetime.date) the last date to collect articles from
def collect_articles(source_domain, start_date=DEFAULT_START_DATE, end_date=date.today()):
    article_list = []

    date_range_list = datespan(start_date=start_date, end_date=end_date)
    for cur_date in date_range_list:
        print("getting", source_domain, "articles from", cur_date)
        data = get_headlines(source_domain, from_date=cur_date, to_date=cur_date)

        articles = data["articles"]
        for article_dict in articles:
            # only use the relevant information from each article
            source = article_dict["source"]["id"]
            headline = article_dict["title"]

            minimal_article_dict = dict()
            minimal_article_dict["source"] = source
            minimal_article_dict["headline"] = headline
            minimal_article_dict["date"] = str(cur_date)
            article_list.append(minimal_article_dict)
    return article_list


# Returns a python list or dictionary that is loaded from a local json file
#   source: (string) the news source for which articles will be loaded
def load_articles(source):
    with open('./' + source + '_headlines.json') as json_file:
        data = json.load(json_file)
        # print("locally read " + source + " data:", data)
        return data


# Saves a python list or dictionary to a json file
#   source: (string) the news source that the json file will be titled after
#   data: (list or dictionary) the data to be saved to a json file
def save_articles(source, data):
    # print("saving " + source + " data:", data)
    json_object = json.dumps(data, indent=4)
    with open(source + "_headlines.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    pass
