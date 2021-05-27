import datetime as dt
import time
from news_clustering import NewsClustering
from news_processor import NewsProcessor
from json_loader import JsonLoader
from stories_visualizer import visualize_trending_stories
from keywords_post_process import KeywordsPostProcessor


if __name__ == "__main__":

    # uncomment the following lines of codes to rerun keywords extraction
    # news_articles = NewsProcessor.load_news_csv(file="data/reuters_headlines.csv")
    # news_providers = NewsProcessor.get_provider_names(news_articles)
    # print("news providers: ", news_providers)
    # t0 = time.time()
    # news_articles = NewsProcessor.extract_keywords_news_list(news_articles)
    # news_articles = [_d for _d in news_articles if _d['kwords']]
    # JsonLoader.save_json(filename="data/reuters_cleaned_with_keywords.json", data=news_articles)
    # t1 = time.time()
    # print("number of articles {}, keywords extraction time {}s".format(len(news_articles), t1-t0))

    KeywordsPostProcessor.post_process_article_keywords()
    news_articles = JsonLoader.load_json("data/reuters_cleaned_with_keywords_post_processed.json")
    for _d in news_articles:
        _d['pubDate'] = dt.datetime.strptime(_d['pubDate'], "%Y-%m-%dT%H:%M:%S")
    news_articles = [_d for _d in news_articles if _d['pubDate'] > dt.datetime(2019, 11, 15)]
    news_clusters = NewsClustering.cluster_news_by_weighted_keywords(news_articles, eps=0.35, max_size=500)
    NewsClustering.get_cluster_id_and_core_sample_news(news_clusters, news_articles)
    not_clustered = [_d for _d in news_articles if _d['cluster_id'] is None]
    if not_clustered:
        print("number of news not clustered {}".format(len(not_clustered)))
        news_articles = [_d for _d in news_articles if _d['cluster_id'] is not None]
    visualize_trending_stories(news_articles)

