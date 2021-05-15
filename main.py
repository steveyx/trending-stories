import datetime as dt
from news_clustering import NewsClustering
from news_processor import NewsProcessor
from json_loader import JsonLoader
from stories_visualizer import visualize_trending_stories


if __name__ == "__main__":

    news_articles = NewsProcessor.load_news_csv(file="data/reuters_headlines.csv")
    news_providers = NewsProcessor.get_provider_names(news_articles)
    print("news providers: ", news_providers)
    news_articles = NewsProcessor.extract_keywords_news_list(news_articles)
    news_articles = [_d for _d in news_articles if _d['kwords']]
    JsonLoader.save_json("data/reuters_cleaned_with_keywords_1.json", news_articles)

    news_articles = JsonLoader.load_json("data/reuters_cleaned_with_keywords_post_processed.json")
    for _d in news_articles:
        _d['pubDate'] = dt.datetime.strptime(_d['pubDate'], "%Y-%m-%dT%H:%M:%S")
    news_articles = [_d for _d in news_articles if _d['pubDate'] > dt.datetime(2020, 5, 15)]
    print("number of loaded articles: {}".format(len(news_articles)))
    news_clusters = NewsClustering.cluster_news_by_weighted_keywords(news_articles, eps=0.25)
    article2cluster_map = {_article['_id']: _c['cluster_id'] for _c in news_clusters for _article in _c['news']}
    for _d in news_articles:
        _d['cluster_id'] = article2cluster_map.get(_d['_id'])
    not_clustered = [_d for _d in news_articles if _d['cluster_id'] is None]
    print("number of news not clustered {}".format(len(not_clustered)))
    if not_clustered:
        news_articles = [_d for _d in news_articles if _d['cluster_id'] is not None]
    visualize_trending_stories(news_articles)
