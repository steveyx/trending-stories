from json_loader import JsonLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class KeywordsPostProcessor:
    replace_keywords = {
        "United States": "US",
        "United Nations": "UN",
        "European Union": "EU",
        "United Kingdom": "UK"
    }

    @staticmethod
    def get_top_keywords_from_articles(filename="data/reuters_cleaned_with_keywords.json",
                                       save_file="data/top_keywords.csv",
                                       top_n=50):
        _articles = JsonLoader.load_json(filename)
        _article_keywords = [a['kwords'] for a in _articles if a.get('kwords')]
        _all_keywords = [[w['keyword'], 1] for a in _article_keywords for w in a]
        _df = pd.DataFrame(_all_keywords, columns=["keyword", "count"])
        _df_g = _df.groupby(by="keyword", as_index=False).agg(
            {"count": sum}
        )
        _df_g.sort_values(by="count", inplace=True, ascending=False)
        _df_g.to_csv(save_file, index=False)
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(y="keyword", x="count", data=_df_g.iloc[:top_n])
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        plt.tight_layout()
        plt.show()

    @classmethod
    def post_process_article_keywords(cls, filename="data/reuters_cleaned_with_keywords.json"):
        _articles = JsonLoader.load_json(filename)
        for a in _articles:
            for _word in a.get('kwords', []):
                for k, v in cls.replace_keywords.items():
                    _word['keyword'] = _word['keyword'].replace(k, v)
        _file_name = filename.split(".json")[0] + "_post_processed.json"
        JsonLoader.save_json(_articles, _file_name)


if __name__ == "__main__":
    KeywordsPostProcessor.get_top_keywords_from_articles()
    KeywordsPostProcessor.post_process_article_keywords()
    KeywordsPostProcessor.get_top_keywords_from_articles(
        filename="data/reuters_cleaned_with_keywords_post_processed.json",
        save_file="data/top_keywords_post_processed.csv")
