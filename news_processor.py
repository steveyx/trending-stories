import pandas as pd
from keywords_extraction import KeywordsExtract
from news_cleaner import NewsCleaner


class NewsProcessor:

    @staticmethod
    def clean_and_extract_keywords(news):
        raw_title = news['title']
        raw_content = news['content']
        cleaned_title, cleaned_content = NewsCleaner.clean_news_article(title=raw_title, content=raw_content)
        df, df_chunk, kwords_dict = KeywordsExtract.extract_keywords(cleaned_title, cleaned_content)
        news["cleanedTitle"] = cleaned_title
        news["cleanedContent"] = cleaned_content
        news["kwords"] = kwords_dict
        print("{}:\n\t{}\n".format(cleaned_title, kwords_dict))
        return news

    @staticmethod
    def extract_keywords_news_list(news_list):
        processed_news = [NewsProcessor.clean_and_extract_keywords(a) for a in news_list]
        return processed_news

    @staticmethod
    def load_news_csv(file):
        _df = pd.read_csv(file).iloc[:]
        if "reuters" in file:
            _df.columns = ["title", "pubDate", "content"]
            _df["_id"] = _df.index.map(lambda x: "reuters_{}".format(x))
            _df["provider"] = "reuters"
            _df["title"] = _df["title"].str.strip()
        _df["pubDate"] = pd.to_datetime(_df["pubDate"])
        _df = _df[_df["title"].str.len() > 1]
        _df.sort_values(by=["pubDate"], inplace=True)
        _dict = _df.to_dict(orient="records")
        return _dict

    @staticmethod
    def get_provider_names(articles):
        _providers = [d['providerName'] if 'providerName' in d else d['provider'] for d in articles]
        return list(set(_providers))

