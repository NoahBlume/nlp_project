from sklearn.cluster import KMeans
import numpy as np
import warnings
from yellowbrick.cluster import KElbowVisualizer


warnings.filterwarnings('ignore')

NUM_CLUSTER_OPTIONS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25]
BASE_LABEL_STRING = ' labels'
BASE_CLUSTER_STRING = ' clusters'

SENTIMENT_LABEL_STRING = 'sentiment'
TFIDF_LABEL_STRING = 'tfidf'


def add_article_labels(labelable_articles, label_type, labels, n_clusters):
    for i in range(len(labelable_articles)):
        article = labelable_articles[i]
        label_dict_key = label_type + BASE_LABEL_STRING
        label_dict = article.get(label_dict_key, dict())
        label_dict[str(n_clusters) + BASE_CLUSTER_STRING] = str((labels[i]))
        article[label_dict_key] = label_dict


def get_tfidf_ndarray(articles):
    vectors = []
    for article in articles:
        tfidf_vector = article['tfidf_vector']
        vectors.append(tfidf_vector)
    return np.array(vectors)


def get_sentiment_ndarray(articles):
    vectors = []
    for article in articles:
        polarity = article['sentiment_polarity']
        subjectivity = article['sentiment_subjectivity']
        vectors.append([polarity, subjectivity])
    return np.array(vectors)


def get_kmeans_labels(vectors, n_clusters):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 25))

    visualizer.fit(vectors)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure


def create_labellable_articles(articles):
    labellable_articles = []
    for article in articles:
        labellable_article = dict()
        labellable_article['source'] = article['source']
        labellable_article['headline'] = article['headline']
        # labellable_article['date'] = article['date']

        labellable_articles.append(labellable_article)
    return labellable_articles


def run_kmeans_and_label_articles(articles):
    sentiment_vectors = get_sentiment_ndarray(articles)
    tfidf_vectors = get_tfidf_ndarray(articles)

    print("running kmeans on sentiment vectors")
    get_kmeans_labels(sentiment_vectors, 0)
    print("running kmeans on tfidf vectors")
    get_kmeans_labels(tfidf_vectors, 0)




def group_headlines_by_label(n_clusters, label_type, labelled_articles):
    buckets = [[] for n in range(n_clusters)]

    for article in labelled_articles:
        label_dict_key = label_type + BASE_LABEL_STRING
        # print("article:", article)
        # print("label dict key:", label_dict_key)
        type_labels = article[label_dict_key]

        cluster_key = str(n_clusters) + BASE_CLUSTER_STRING

        try:
            n_label = int(type_labels[cluster_key])
        except KeyError:
            # ignore key errors
            continue
        buckets[n_label].append((article['source'], article['headline']))
    return buckets


def create_custer_groups(labelled_articles, include_headlines=True):
    cluster_groups = dict()

    sentiment_groups = dict()
    tfidf_groups = dict()

    for n in NUM_CLUSTER_OPTIONS:
        all_sentiment_cluster_stats = dict()
        sentiment_cluster_buckets = group_headlines_by_label(n, SENTIMENT_LABEL_STRING, labelled_articles)
        sentiment_cluster_group = dict()

        sentiment_cluster_group["cluster stats"] = all_sentiment_cluster_stats
        for i in range(len(sentiment_cluster_buckets)):
            stats = dict()
            bucket_count = 0
            bucket_dict = dict()
            for source, headline in sentiment_cluster_buckets[i]:
                bucket_count += 1
                source_count = stats.get(source + ' count', 0)
                stats[source + ' count'] = source_count + 1

                cluster_total_source_count = all_sentiment_cluster_stats.get(source + ' count', 0)
                all_sentiment_cluster_stats[source + ' count'] = cluster_total_source_count + 1

            stats["total count"] = bucket_count

            new_entries = []
            for key, item in stats.items():
                new_key = key.replace("count", "percent")
                percent = 0
                if bucket_count > 0:
                    percent = 100 * item / bucket_count
                new_entries.append((new_key, percent))
            for key, item in new_entries:
                stats[key] = item

            del stats["total percent"]
            bucket_dict["source counts"] = stats
            if include_headlines:
                bucket_dict['headlines'] = sentiment_cluster_buckets[i]
            sentiment_cluster_group[str(i)] = bucket_dict

        # calculate total stats
        newer_entries = []
        for key, item in all_sentiment_cluster_stats.items():
            total_source_count = item
            for i in range(len(sentiment_cluster_buckets)):
                cluster_key = str(i)
                per_cluster_count = sentiment_cluster_group[cluster_key]["source counts"].get(key, 0)
                cluster_percentage = 100 * per_cluster_count / total_source_count
                newer_entries.append((key.replace("count", "percent") + " - cluster " + cluster_key, cluster_percentage))

        for key, item in newer_entries:
            all_sentiment_cluster_stats[key] = item

        sentiment_groups[str(n) + ' ' + SENTIMENT_LABEL_STRING + BASE_CLUSTER_STRING] = sentiment_cluster_group


        all_tfidf_cluster_stats = dict()
        tfidf_cluster_buckets = group_headlines_by_label(n, TFIDF_LABEL_STRING, labelled_articles)
        tfidf_cluster_group = dict()

        tfidf_cluster_group["cluster stats"] = all_tfidf_cluster_stats
        for i in range(len(tfidf_cluster_buckets)):
            stats = dict()
            bucket_count = 0
            bucket_dict = dict()
            for source, headline in tfidf_cluster_buckets[i]:
                bucket_count += 1
                source_count = stats.get(source + ' count', 0)
                stats[source + ' count'] = source_count + 1

                cluster_total_source_count = all_tfidf_cluster_stats.get(source + ' count', 0)
                all_tfidf_cluster_stats[source + ' count'] = cluster_total_source_count + 1

            stats["total count"] = bucket_count

            new_entries = []
            for key, item in stats.items():
                new_key = key.replace("count", "percent")
                percent = 0
                if bucket_count > 0:
                    percent = 100 * item / bucket_count
                new_entries.append((new_key, percent))
            for key, item in new_entries:
                stats[key] = item

            del stats["total percent"]
            bucket_dict["source counts"] = stats
            if include_headlines:
                bucket_dict['headlines'] = tfidf_cluster_buckets[i]
            tfidf_cluster_group[str(i)] = bucket_dict

        # calculate total stats
        newer_entries = []
        for key, item in all_tfidf_cluster_stats.items():
            total_source_count = item
            for i in range(len(tfidf_cluster_buckets)):
                cluster_key = str(i)
                per_cluster_count = tfidf_cluster_group[cluster_key]["source counts"].get(key, 0)
                cluster_percentage = 100 * per_cluster_count / total_source_count
                newer_entries.append(
                    (key.replace("count", "percent") + " - cluster " + cluster_key, cluster_percentage))

        for key, item in newer_entries:
            all_tfidf_cluster_stats[key] = item

        tfidf_groups[str(n) + ' ' + TFIDF_LABEL_STRING + BASE_CLUSTER_STRING] = tfidf_cluster_group

    cluster_groups[SENTIMENT_LABEL_STRING + BASE_LABEL_STRING] = sentiment_groups
    cluster_groups[TFIDF_LABEL_STRING + BASE_LABEL_STRING] = tfidf_groups
    return cluster_groups


if __name__ == "__main__":
    pass

    # example commands below:
#     fox_articles = load_articles("foxnews_tfidf_added")
#     msnbc_articles = load_articles("msnbc_tfidf_added")
    
#     combined_articles = fox_articles + msnbc_articles
#     labelled_combined_articles = run_kmeans_and_label_articles(combined_articles)
    
#     combined_articles_by_cluster = create_custer_groups(labelled_combined_articles)
#     combined_articles_by_cluster_no_headlines = create_custer_groups(labelled_combined_articles, include_headlines=False)
    
#     save_articles("combined_grouped_by_cluster", combined_articles_by_cluster)
#     save_articles("combined_grouped_by_cluster_without_headlines", combined_articles_by_cluster_no_headlines)
