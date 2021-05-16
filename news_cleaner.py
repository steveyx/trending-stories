import re
import unicodedata


class NewsCleaner:
    non_en_chars = {
        "’": "'",
        "‘": "'"
    }
    remove_from_title = ["BREAKING:", "[\-:] report", "[\-:] sources",  "[\-:] source", "source says", "Exclusive\:",
                         "Factbox\:", "Timeline[\-:]",  "Instant View\:", "Explainer\:", ": Bloomberg",
                         ": WSJ"]
    remove_if_start_with = ['close breaking news']
    replace_if_contain = ['click here to see']

    @staticmethod
    def clean_news_article(title="", content=""):
        _c_title = NewsCleaner.clean_title(title)
        _c_content = NewsCleaner.clean_content(content)
        return _c_title, _c_content

    @staticmethod
    def clean_title(raw_title):
        txt = NewsCleaner.remove_non_en_chars(raw_title)
        txt = NewsCleaner.replace_abbreviation(txt)
        p = "|".join(NewsCleaner.remove_from_title)
        txt = re.sub(p, '', txt)
        txt = re.sub(r'\s+', " ", txt).strip()
        return txt

    @staticmethod
    def clean_content(raw_content):
        if not raw_content:
            return ""
        remove_if_start_with = NewsCleaner.remove_if_start_with
        replace_if_contain = NewsCleaner.replace_if_contain
        txt = NewsCleaner.remove_non_en_chars(raw_content)
        txt = NewsCleaner.replace_abbreviation(txt)
        lines = txt.split("\n")
        # remove short sentence
        clean1 = [l for l in lines if len(l) > 30]
        clean2 = [l for l in clean1 if len(l.split(" ")) > 3]
        # remove the words if the sentence starts with certain pattern
        p_start = r"^ *" + r"|^ *".join(remove_if_start_with)
        clean3 = [l for l in clean2 if re.match(p_start, l) is None]
        # remove the words if the sentence contains certain pattern
        p_contain = r"|".join(replace_if_contain)
        clean4 = [re.sub(p_contain, '', l) for l in clean3]
        clean5 = " ".join(clean4)
        return re.sub(r'\s+', ' ', clean5).strip()

    @staticmethod
    def remove_non_en_chars(txt):
        # remove non english characters
        txt = NewsCleaner.convert_latin_chars(txt)
        for char in NewsCleaner.non_en_chars.keys():
            txt = re.sub(char, NewsCleaner.non_en_chars[char], txt)
        txt = re.sub(r'[^\x00-\x7F]+', ' ', txt)
        return txt

    @staticmethod
    def convert_latin_chars(txt):
        # convert latin characters
        return ''.join(char for char in unicodedata.normalize('NFKD', txt) if unicodedata.category(char) != 'Mn')

    @staticmethod
    def replace_abbreviation(txt):
        words = {'U.S.': 'US'}
        for w in words.keys():
            txt = re.sub(w, words[w], txt)
        return txt


if __name__ == "__main__":
    title = "Facebook's Zuckerberg to testify before Congress: source"
    content = "Facebook Inc Chief Executive Mark Zuckerberg plans to testify before U.S. Congress, a source briefed " \
              "on the matter said on Tuesday, as he bows to pressure from lawmakers insisting he explain how 50 " \
              "million users' data ended up in the hands of a political consultancy."

    cleaned_title, cleaned_content = NewsCleaner.clean_news_article(title=title, content=content)
    print(cleaned_title)
    print(cleaned_content)
