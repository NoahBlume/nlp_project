from headline_retriever import load_articles, save_articles
from pre_processor import pre_process
from tfidf_vectorizer import fit_vectorizer, add_tfidf_vectors
from kmeans_clusterer import run_kmeans_and_label_articles, create_cluster_groups

if __name__ == "__main__":
    print("loading articles")
    fox_articles = load_articles("headlines/foxnews_headlines")
    msnbc_articles = load_articles("headlines/msnbc_headlines")

    print("pre-processing articles")
    pre_process(fox_articles)
    pre_process(msnbc_articles)

    print("saving pre-processed articles (this may take ~10 minutes)")
    save_articles("foxnews_pre_processed", fox_articles)
    save_articles("msnbc_pre_processed", msnbc_articles)

    print("fitting tfidf vectorizer for article vocabulary")
    fit_vectorizer([fox_articles, msnbc_articles])

    print("adding tfidf vectors to articles")
    add_tfidf_vectors(fox_articles)
    add_tfidf_vectors(msnbc_articles)

    print("saving articles with tfidf vectors (this may take ~30 seconds - the files are somewhat large)")
    save_articles("foxnews_tfidf_added", fox_articles)
    save_articles("msnbc_tfidf_added", msnbc_articles)

    print("labelling articles with k means (this may take ~15 minutes)")
    combined_articles = fox_articles + msnbc_articles
    labelled_combined_articles = run_kmeans_and_label_articles(combined_articles)

    print("organizing articles into groups based on their clusters")
    combined_articles_by_cluster = create_cluster_groups(labelled_combined_articles)
    combined_articles_by_cluster_no_headlines = create_cluster_groups(labelled_combined_articles, include_headlines=False)

    print("saving grouped and clustered articles")
    save_articles("combined_grouped_by_cluster", combined_articles_by_cluster)
    save_articles("combined_grouped_by_cluster_without_headlines", combined_articles_by_cluster_no_headlines)

    print("DONE!")
