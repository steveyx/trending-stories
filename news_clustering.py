import re
import pandas as pd
from sklearn.cluster import DBSCAN
from cosine_similarity import calculate_cosine_similarity
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
    def cluster_news_by_weighted_keywords(cls, news_articles, eps=0.25, by='kwords', max_size=200):
        if not news_articles:
            return []
        lists = cls.get_keywords_lists(news_articles, by=by)
        joint_kwords_list = lists["joint_kwords_list"]
        news_list = lists["news_list"]
        m_sim = calculate_cosine_similarity(joint_kwords_list)
        m_distance = 1.0 - m_sim
        m_distance[m_distance < 0] = 0.0
        results = cls.db_scan_recursive(dist_matrix=m_distance, eps=eps, max_size=max_size)
        c = [[results[k]["cluster"], results[k]["core_index"]] for k in sorted(results)]
        df = pd.DataFrame(c, columns=['cluster', "core_news"])
        df['news'] = news_list
        # df1 = df.groupby(['cluster'])  ['_id'].apply(list).to_frame(name='news')
        df1 = df.groupby(['cluster']).agg({
            "news": list,
            "core_news": lambda x: news_list[list(x)[0]]
        })
        df1['count'] = df1['news'].apply(lambda x: len(x))
        _clusters = [{"cluster_id": str(idx), "news": x['news'], "cluster_size": len(x['news']),
                      "core_news": x['core_news'], "eps": eps}
                     for idx, x in df1.iterrows()]
        for _c in _clusters:
            _cluster_kwords = cls.get_sorted_keywords_for_cluster(_c["news"])
            _c['cluster_keywords'] = _cluster_kwords
        return _clusters

    @classmethod
    def cluster_news_simple(cls, news_articles, eps=0.2):
        """
        small eps clustering for fine tuning
        """
        if not news_articles:
            return []
        model = DBSCAN(eps, min_samples=1, metric="precomputed", algorithm='brute')
        joint_kwords_list = [" ".join(j["keywords"]) for j in news_articles]
        article_ids_list = [j["cluster_id"] for j in news_articles]
        m_sim = calculate_cosine_similarity(joint_kwords_list)
        m_distance = 1.0 - m_sim
        m_distance[m_distance < 0] = 0.0
        model = cls.model_fit(model=model, dist_matrix=m_distance)
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

    @classmethod
    def model_fit(cls, model=None, dist_matrix=None):
        model.fit(dist_matrix)
        return model

    @classmethod
    def db_scan_recursive(cls, dist_matrix=None, eps=0.25, max_size=200):
        model = DBSCAN(eps, min_samples=1, metric="precomputed", algorithm='brute')
        model.fit(dist_matrix)
        c = model.labels_.reshape((-1, 1))
        core_indices = model.core_sample_indices_.ravel()
        df = pd.DataFrame(c, columns=['cluster'])
        df['news_indices'] = df.index
        df['cluster_size'] = 1
        df1 = df.groupby(['cluster'], as_index=False).agg({
            'news_indices': list,
            'cluster_size': sum
        })
        # results is a list of {news_index: {"cluster": label, "core_index": core_indices[i]} }
        results = {i: {"cluster": label, "core_index": core_indices[i]}
                   for i, label in enumerate(model.labels_)}
        for idx, row in df1.iterrows():
            if row["cluster_size"] > max_size and eps > 0.1:
                indices = row['news_indices']
                idx_mesh = np.meshgrid(indices, indices, sparse=False, indexing='ij')
                sub_matrix = dist_matrix[tuple(idx_mesh)]
                next_results = cls.db_scan_recursive(dist_matrix=sub_matrix,
                                                     eps=eps-0.05, max_size=max_size)
                label_max = max([v["cluster"] for v in results.values()]) + 1
                for k, v in next_results.items():
                    results[indices[k]] = {"cluster": label_max + v["cluster"], "core_index": indices[v["core_index"]]}
        return results

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

    @classmethod
    def get_cluster_id_and_core_sample_news(cls, news_clusters, news_articles, top_n=2, num=50):
        article2cluster_map = {_article['_id']: {"cluster_id": _c['cluster_id'], "core_news": _c['core_news']}
                               for _c in news_clusters for _article in _c['news']}
        _sizes = [c["cluster_size"] for c in news_clusters]
        _top_cluster_ids = []
        core_news = {}
        if len(_sizes) > top_n:
            _top_n_indices = np.argsort(_sizes)[-top_n:][::-1]
            _top_cluster_ids = [news_clusters[i]["cluster_id"] for i in _top_n_indices]
            for _c_idx in _top_n_indices:
                c_news = news_clusters[_c_idx]["news"]
                c_news.sort(key=lambda x: x["pubDate"])
                for i in range(0, len(c_news), num):
                    end = len(c_news) if i+num*1.5 >= len(c_news) else i + num
                    news_s = c_news[i: end]
                    lists = cls.get_keywords_lists(news_s, by="kwords")
                    joint_kwords_list = lists["joint_kwords_list"]
                    m_sim = calculate_cosine_similarity(joint_kwords_list)
                    max_idx = np.argmax(m_sim.sum(axis=1))
                    for n in news_s:
                        core_news[n['_id']] = news_s[max_idx]
        for _d in news_articles:
            _d['cluster_id'] = article2cluster_map.get(_d['_id'], {}).get("cluster_id")
            if _d["_id"] in core_news:
                _d["core_news"] = core_news[_d["_id"]]
            else:
                _d["core_news"] = article2cluster_map.get(_d['_id'], {}).get("core_news")
