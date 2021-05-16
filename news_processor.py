import pandas as pd
from keywords_extraction import KeywordsExtract
from news_cleaner import NewsCleaner
import datetime as dt


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
        _df["pubDate"] = _df["pubDate"].apply(lambda x: dt.datetime.strftime(x, "%Y-%m-%dT%H:%M:%S"))
        _dict = _df.to_dict(orient="records")
        return _dict

    @staticmethod
    def get_provider_names(articles):
        _providers = [d['providerName'] if 'providerName' in d else d['provider'] for d in articles]
        return list(set(_providers))


if __name__ == "__main__":
    articles = [
        {
            "title": "SoftBank mulls $1 billion investment in Chinese truck-hailing firm: WSJ",
            "pubDate": "2018-03-26T00:00:00",
            "content": "Japan's SoftBank Group Corp  is looking to invest $1 billion in Chinese "
                       "truck-hailing company Manbang Group, the Wall Street Journal reported on "
                       "Monday, citing people familiar with the matter."},
        {
            "title": "Exclusive: S&P Global cuts top sovereign analysts as part of broader cull",
            "pubDate": "2018-03-26T00:00:00",
            "content": "The world's biggest credit ratings agency S&P Global has cut more than five of its most senior "
                       "sovereign analysts as part of a broader reduction of in excess of 100 staff, a source "
                       "told Reuters."
        },
        {
            "title": "Ex-Deutsche Telekom boss favourite to be next Airbus chairman: report",
            "pubDate": "2018-03-27T00:00:00",
            "content": "Former Deutsche Telekom  chief executive Rene Obermann is the favourite to take "
                       "over as chairman at European aerospace company Airbus , the Handelsblatt newspaper "
                       "reported on Tuesday, citing German government and diplomatic sources."
        },
        {
            "title": "Exclusive: Salesforce in advanced talks to buy MuleSoft - sources",
            "pubDate": "2018-03-20T00:00:00",
            "content": "Salesforce.com Inc  is in advanced discussions to acquire U.S. software maker MuleSoft Inc , "
                       "people familiar with the matter told Reuters on Tuesday, as it looks to expand its "
                       "offerings beyond customer relationship management software.",
        },
        {
            "title": "Take Five: World markets themes for the week ahead",
            "pubDate": "2018-03-29T00:00:00",
            "content": "Following are five big themes likely to dominate thinking of investors and traders in "
                       "the coming week and the Reuters stories related to them.",
        },
        {
            "title": "Italy Antitrust opens probe into Facebook's collection, use of data",
            "pubDate": "2018-04-06T00:00:00",
            "content": "Italy's Antitrust Authority has opened a probe into possible incorrect commercial "
                       "practices by Facebook in its treatment of user data, the agency said in a statement "
                       "on Friday.",
        }
    ]
    result = NewsProcessor.extract_keywords_news_list(articles)
    from pprint import pprint

    pprint(result)
