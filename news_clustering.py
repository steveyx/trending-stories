import re
import pandas as pd
from sklearn.cluster import DBSCAN
from cosine_similarity import get_cosine_sim
import numpy as np


np.random.seed(2)


class NewsClustering:

    @classmethod
    def get_weighted_keywords(cls, keywords_list):
        k_dict = [x['kwords'] for x in keywords_list]
        keywords_counted = [[(v['keyword']+' ')*v['weight'] for v in k] for k in k_dict]
        weighted_kwords = cls.split_keywords(keywords_counted)
        return weighted_kwords

    @classmethod
    def split_keywords(cls, keywords_counted):
        a = [" ".join(x) for x in keywords_counted]
        doc_kwords = [re.split('___|,|\-|h=/|\\|\*| ', k.lower()) for k in a]
        doc_kwords = [[v for v in k if len(v) > 0] for k in doc_kwords]
        return doc_kwords

    @classmethod
    def cluster_news_by_weighted_keywords(cls, news_articles, eps, by='kwords'):
        if news_articles and len(news_articles) > 0:
            if not eps:
                eps = 0.4
            model = DBSCAN(eps, min_samples=1, metric="precomputed", algorithm='brute')
            lists = cls.get_keywords_lists(news_articles, by=by)
            joint_kwords_list = lists["joint_kwords_list"]
            news_list = lists["news_list"]
            m_sim = get_cosine_sim(joint_kwords_list)
            reverse_m_sim = 1.0 - m_sim
            reverse_m_sim[reverse_m_sim < 0] = 0.0
            model = cls.model_fit(model=model, dist_matrix=reverse_m_sim, weighted_kwords_list=lists["weighted_list"])
            c = model.labels_.reshape((-1, 1))
            df = pd.DataFrame(c, columns=['cluster'])
            df['_id'] = news_list
            df1 = df.groupby(['cluster'])['_id'].apply(list).to_frame(name='news')
            df1['count'] = df1['news'].apply(lambda x: len(x))
            _clusters = [{"cluster_id": idx, "news": x['news'], "cluster_size": len(x['news']), "eps": eps}
                         for idx, x in df1.iterrows()]
            for _c in _clusters:
                _cluster_kwords = cls.get_sorted_keywords_for_cluster(_c["news"])
                _c['cluster_keywords'] = _cluster_kwords
            return _clusters
        else:
            return None

    @classmethod
    def cluster_news_simple(cls, news_articles, eps=0.2):
        """
        small eps clustering for fine tuning
        """
        if news_articles and len(news_articles) > 0:
            if not eps:
                eps = 0.2
            model = DBSCAN(eps, min_samples=1, metric="precomputed", algorithm='brute')
            joint_kwords_list = [" ".join(j["keywords"]) for j in news_articles]
            article_ids_list = [j["cluster_id"] for j in news_articles]
            m_sim = get_cosine_sim(joint_kwords_list)
            reverse_m_sim = 1.0 - m_sim
            reverse_m_sim[reverse_m_sim < 0] = 0.0
            model = cls.model_fit(model=model, dist_matrix=reverse_m_sim, weighted_kwords_list=[])
            c = model.labels_.reshape((-1, 1))
            df = pd.DataFrame(c, columns=['cluster'])
            df['_id'] = article_ids_list
            df['cluster_id'] = article_ids_list
            df["cluster_size"] = 1
            df["news_indices"] = df.index
            df1 = df.groupby(['cluster'], as_index=False).agg({
                '_id': list,
                'news_indices': list,
                'cluster_id': lambda x: list(x)[0],
                'cluster_size': sum
            }).rename(columns={"_id": "news_ids"})
            _clusters = [{"cluster_id": x["cluster_id"],
                          "news_indices": x['news_indices'],
                          "news_ids": x['news_ids'], "cluster_size": x['cluster_size'], "eps": eps}
                         for idx, x in df1.iterrows()]
            return _clusters
        else:
            return None

    @classmethod
    def model_fit(cls, model=None, dist_matrix=None, weighted_kwords_list=[]):
        if dist_matrix is None:
            str_list = [' '.join(x) for x in weighted_kwords_list]
            dist_matrix = 1 - get_cosine_sim(str_list)
        model.fit(dist_matrix)
        return model

    @classmethod
    def get_keywords_lists(cls, news_article_list, by='kwords'):
        d_kwords = [{'kwords': k.get(by, []), '_id': k} for k in news_article_list]
        w_kwords = cls.get_weighted_keywords(d_kwords)
        kwords = [[w['keyword'] for w in k.get('kwords', [])] for k in d_kwords]
        doc_k = [{'_id': k['_id'], "kwords": kwords[i], "weighted_kwords": w_kwords[i]} for i, k in enumerate(d_kwords)
                 if (len(kwords[i]) > 0 & len(w_kwords[i]))]
        weighted_kwords_list = [k['weighted_kwords'] for k in doc_k]
        joint_kwords_list = [' '.join(x) for x in weighted_kwords_list]
        news_list = [k['_id'] for k in doc_k]
        return {"joint_kwords_list": joint_kwords_list, "news_list": news_list, "weighted_list": weighted_kwords_list}

    @classmethod
    def get_sorted_keywords_for_cluster(cls, cluster_news_list):
        d_kwords = [k.get('kwords', []) for k in cluster_news_list]
        doc_k_list = []
        for doc in d_kwords:
            if len(doc) == 0:
                continue
            for keyword in doc:
                words = re.split('___|,|\-|h=/|\\|\*| ', keyword['keyword'])
                doc_k_list += [[w.lower(), keyword['weight']] for w in words if len(w) > 1]
        if len(doc_k_list) > 0:
            df = pd.DataFrame(doc_k_list, columns=['word', 'weight'])
            df = df.groupby(by='word', as_index=False)['weight'].sum().sort_values(by='weight', ascending=False)\
                .reset_index(drop=True)
            df['percentage'] = df['weight']/df['weight'].sum()
            results = [[x['word'], round(x['percentage'], 4)] for idx, x in df.iterrows()]
        else:
            results = []
        return results

