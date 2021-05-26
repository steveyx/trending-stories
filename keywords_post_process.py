from json_loader import JsonLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100


class KeywordsPostProcessor:
    keywords_linking_table = {
        "United States": "US",
        "United Nations": "UN",
        "European Union": "EU",
        "United Kingdom": "UK",
        "European Central Bank": "ECB",
        "World Trade Organization": "WTO",
        "initial public offering": "IPO",
        "International Monetary Fund": "IMF",
        "chief executive officer": "CEO",
        "chief financial officer": "CFO",
        "JP Morgan": "JPMorgan",
        "Johnson & Johnson": "J&J",
        "Federal Reserve": "Fed",
        "Amazoncom Inc": "Amazon",
        "Amazoncom": "Amazon",
        "General Motors Co": "GM",
        "General Motors": "GM",
        "General Motor": "GM",
        "Wall St": "Wall Street",
        "Boeing Co": "Boeing",
        "US Federal Reserve": "US Fed",
        "Uber Technologies Inc": "Uber",
        "chief executive": "CEO",
        "Federal Aviation Administration": "FAA",
        "Wall Street Journal": "WSJ",
        "General Electric": "GE",
        "General Electric Co": "GE",
        "Federal Communications Commission": "FCC"
    }

    @classmethod
    def get_top_keywords_from_articles(cls, filename="data/reuters_cleaned_with_keywords.json",
                                       save_file="data/top_keywords.csv",
                                       table_plot=True,
                                       top_n=34):
        _articles = JsonLoader.load_json(filename)
        _article_keywords = [a['kwords'] for a in _articles if a.get('kwords')]
        _all_keywords = [[w['keyword'], 1] for a in _article_keywords for w in a]
        _df = pd.DataFrame(_all_keywords, columns=["Keyword", "Count"])
        _df_g = _df.groupby(by="Keyword", as_index=False).agg(
            {"Count": sum}
        )
        _df_g.sort_values(by="Count", inplace=True, ascending=False)
        _df_g.reset_index(drop=True, inplace=True)
        _df_g.to_csv(save_file, index=False)
        if table_plot:
            words_and_abbr = []
            for k, v in cls.keywords_linking_table.items():
                _df_t = _df_g.loc[:top_n * 3]
                _found = _df_t[_df_t["Keyword"].isin([k, v])]
                if len(_found) == 2:
                    _indices = _found.index.tolist()
                    words_and_abbr.append(_indices)
            fig, axs = plt.subplots(1, 3, figsize=(8, 6))
            tables = []
            for i, ax in enumerate(axs):
                ax.axis('off')
                _s, _e = i * top_n, (i+1) * top_n
                tab = ax.table(cellText=_df_g.iloc[_s:_e][["Keyword", "Count"]].values,
                               cellLoc='center', rowLoc='center',
                               colWidths=[0.7, 0.3],
                               colLabels=["Keyword", 'Count'], rowLabels=_df_g.index[_s:_e].tolist(), loc="center",
                               bbox=[0.05, 0.02, .9, 0.95])
                tables.append(tab)
            for tab in tables:
                scalex, scaley = 1, 1
                tab.scale(scalex, scaley)
                tab.auto_set_font_size(False)
                tab.set_fontsize(7)
                for key, cell in tab.get_celld().items():
                    cell.set_linewidth(0)
            for k, words in enumerate(words_and_abbr):
                color = plt.cm.jet(7 * (k+1) * 0.02 % 1)
                for w in words:
                    row, t_i = w % top_n, int(w/top_n)
                    tables[t_i][(row+1, 0)].set_facecolor(color)
            plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2, rect=(0.05, 0.05, 0.95, 0.95))
            plt.subplots_adjust(wspace=0.4)
            plt.savefig("data/entities_linking.png", dpi=300)
        else:
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(y="Keyword", x="Count", data=_df_g.iloc[:top_n])
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
        plt.show()

    @classmethod
    def post_process_article_keywords(cls, filename="data/reuters_cleaned_with_keywords.json"):
        _articles = JsonLoader.load_json(filename)
        for a in _articles:
            kwords_dict = {_word['keyword']: _word['weight'] for _word in a.get('kwords', [])}
            for k, v in cls.keywords_linking_table.items():
                if k in kwords_dict or k.lower() in kwords_dict:
                    if v in kwords_dict:
                        kwords_dict[v] = kwords_dict[v] + kwords_dict[k]
                    else:
                        kwords_dict[v] = kwords_dict[k]
                    del kwords_dict[k]
                    a["kwords"] = [{"keyword": k, "weight": v} for k, v in kwords_dict.items()]
        _file_name = filename.split(".json")[0] + "_post_processed.json"
        JsonLoader.save_json(data=_articles, filename=_file_name)


if __name__ == "__main__":
    KeywordsPostProcessor.get_top_keywords_from_articles()
    KeywordsPostProcessor.post_process_article_keywords()
    KeywordsPostProcessor.get_top_keywords_from_articles(
        filename="data/reuters_cleaned_with_keywords_post_processed.json",
        save_file="data/top_keywords_post_processed.csv")
