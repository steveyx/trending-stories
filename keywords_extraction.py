import re
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from cosine_similarity import calculate_cosine_similarity
import calendar as cd


class KeywordsExtract:
    MODEL = spacy.load("en_core_web_lg")
    allow_types = ['PERSON', 'GPE', 'ORG', 'NORP', 'LOC', 'FAC', 'WORK_OF_ART', 'EVENT', 'LAW', 'PRODUCT']
    remove_words = ['new', 'time', 'matter', 'source', 'people', 'story', 'reuters story']
    remove_entities = ['REUTERS', 'Reuters', 'Thomson Reuters', 'CNBC']
    months = [cd.month_name[i] for i in range(1, 13)] + [cd.month_abbr[i] for i in range(1, 13)]
    lookups = Lookups()
    lemma_keep = ["data"]
    lemma_exc = MODEL.vocab.lookups.get_table("lemma_exc")
    for w in lemma_keep:
        del lemma_exc[MODEL.vocab.strings["noun"]][w]
    lookups.add_table("lemma_exc", lemma_exc)
    lookups.add_table("lemma_rules", MODEL.vocab.lookups.get_table("lemma_rules"))
    lookups.add_table("lemma_index", MODEL.vocab.lookups.get_table("lemma_index"))
    lemmatizer = Lemmatizer(lookups)

    @classmethod
    def extract_keywords(cls, title, content, title_entity_weight=4, title_noun_chunk_weight=2):
        """
        extract keywords from title and content (or description), with more weights in title entities
        and title noun chunks
        """
        title_txt = title + ". "
        full_content = title_txt + content
        ner_doc = cls.MODEL(full_content)
        df_ent_grp, _ = cls.extract_entities(ner_doc, title=title_txt, title_weight=title_entity_weight)
        df_chunk_grp, _ = cls.extract_noun_chunks(ner_doc, title=title_txt, title_weight=title_noun_chunk_weight)
        df_keywords = cls.filter_keywords_by_score(df_ent_grp, score_threshold=0.05, max_ents=10)
        df_chunks = cls.filter_keywords_by_score(df_chunk_grp, score_threshold=0.05, max_ents=10)
        if not df_chunks.empty:
            df_keywords = df_keywords.append(df_chunks, ignore_index=True, sort=False)
            df_keywords.sort_values(by=['total_score', 'start'], inplace=True, ascending=[False, True])
            df_keywords.drop_duplicates(subset=['entity'], inplace=True)
            df_keywords.drop_duplicates(subset=['lemma'], inplace=True)
        keywords = [{'keyword': str(r['entity']), 'weight': int(r['total_score'])} for _, r in df_keywords.iterrows()]
        return df_ent_grp, df_chunk_grp, keywords

    @classmethod
    def extract_entities(cls, ner_document, title="", title_weight=4):
        ents = [ent for ent in ner_document.ents if ent.label_ in cls.allow_types]
        ents = [cls.trim_tags(ent, for_type='tag') for ent in ents]
        ents = [cls.remove_email_and_punctuation(ent) for ent in ents]
        ents = [ent for ent in ents if len(ent) > 0]
        ent_list = [[cls.lemma_last_word(ent), ent.start_char, ent.end_char, ent.label_, ent.lemma_] for ent in ents]
        cols = ['entity', 'start', 'end', 'label', 'lemma']
        df = pd.DataFrame(ent_list, columns=cols)
        df = cls.filter_keywords(df)
        df['ent_type'] = 'entity'
        df['weight'] = 1
        if not df.empty:
            df['entity'] = df['entity'].str.strip()
            df['weight'] = df['start'].apply(lambda x: title_weight if (x < len(title)) else 1)
        df_grp, df = cls.filter_keywords_and_calculate_weight(df)
        if not df_grp.empty:
            df_grp = cls.merge_keywords_by_similarity(df_grp)
        return df_grp, df

    @classmethod
    def extract_noun_chunks(cls, ner_document, title="", title_weight=2):
        chunks = [ch for ch in list(ner_document.noun_chunks) if (ch.root.ent_type_ == '')]
        chunks = [ch for ch in chunks if len(ch) > 0]
        chunks = [cls.trim_tags(ch) for ch in chunks]
        chunks = [cls.trim_stop_words(ch) for ch in chunks]
        chunks = [cls.remove_email_and_punctuation(ch) for ch in chunks]
        chunks = [cls.trim_entities(ch) for ch in chunks]
        chunks = [ch for ch in chunks if len(ch) > 0]
        chunks_list = [[cls.lemma_last_word(ch), ch.start_char, ch.end_char, ch.label_, ch.lemma_] for ch in chunks]
        cols = ['entity', 'start', 'end', 'label', 'lemma']
        df = pd.DataFrame(chunks_list, columns=cols)
        df = cls.filter_keywords(df)
        df['ent_type'] = 'noun_chunk'
        df['weight'] = 1
        if not df.empty:
            df['entity'] = df['entity'].str.strip()
            df = df[(df['entity'].str.len() > 3) | df['entity'].str.isupper()]
            df['weight'] = df['start'].apply(lambda x: title_weight if (x < len(title)) else 1)
        df_grp, df = cls.filter_keywords_and_calculate_weight(df)
        if not df_grp.empty:
            df_grp = cls.merge_keywords_by_similarity(df_grp)
        return df_grp, df

    @classmethod
    def filter_keywords_and_calculate_weight(cls, df):
        _months = cls.months
        if not df.empty:
            # remove unwanted entities
            df = df[~df['entity'].isin(cls.remove_entities)]
            # remove misclassified dates
            df = df[df['entity'].apply(lambda x: re.search("\s+\d+|".join(_months) + "\s+\d+", str(x)) is None)]
            if df.empty:
                return pd.DataFrame(), df
            df1 = df.groupby(['entity'])[['weight']].sum()
            df0 = df.drop_duplicates(subset=['entity']).copy()
            if "weight" in df0.columns:
                df0.drop(['weight'], axis=1, inplace=True)
            df_merge = pd.merge(df0, df1, left_on='entity', right_index=True)
        else:
            return pd.DataFrame(), df
        df_merge.sort_values(by=['weight', 'start'], inplace=True, ascending=[False, True])
        df_merge = df_merge.reset_index(drop=True)
        return df_merge, df

    @classmethod
    def filter_keywords(cls, df):
        if not df.empty:
            # remove special characters
            df['entity'] = df['entity'].apply(lambda x: re.sub('\.|[\-\'\$\/\\\*\+\|\^\#\@\~\`]{2,}', '', str(x)))
            df = df[df['entity'].apply(lambda x: re.search('\(|\)|\[|\]|\"|\:|\{|\}|\^|\*|\;|\~|\|', str(x)) is None)]
            # remove unwanted words
            if not df.empty:
                df = df[df['entity'].apply(lambda x: x.lower() not in cls.remove_words)]
            # remove too long entities
            if not df.empty:
                df = df[df['entity'].apply(lambda x: len(str(x)) < 35)]
            # remove too short entities
            if not df.empty:
                df = df[df['entity'].apply(lambda x: len(str(x)) > 1)]
        return df

    @staticmethod
    def trim_tags(s, for_type='chunk', trim_tags=['PDT', 'DT', 'IN', 'CC'], punctuation=[',', '\'']):
        if len(s) < 1:
            return s
        s1 = s
        if (for_type == 'chunk') and (re.search('|'.join(punctuation), s1.text) is not None):
            for i in range(len(s1) - 1, -1, -1):
                if s1[i].text in punctuation:
                    s1 = s1[i + 1:]
                    break
        if len(s1) > 0:
            s1 = s1[1:] if (s1[0].tag_ in trim_tags) else s1
        if len(s1) > 1:
            s1 = s1[:-1] if (s1[-1].tag_ in trim_tags) else s1
        return s1

    @staticmethod
    def lemma_last_word(s):
        if s[-1].tag_ in ['NNS', 'NNPS']:
            lemma = KeywordsExtract.lemmatizer(s[-1].text, 'NOUN')[0]
            txt = lemma.title() if s[-1].text.istitle() else lemma
            if len(s) > 1:
                txt = s[:-1].text + " " + txt
        else:
            txt = s.text
        return txt

    @staticmethod
    def filter_keywords_by_score(df, score_threshold=0.1, max_ents=50, top_n=5):
        if df.empty:
            return pd.DataFrame()
        elif len(df) > max_ents:
            df = df[:max_ents]
        cond1 = df['total_score'] / df['total_score'].sum() > score_threshold
        cond2 = df['total_score'] > 0
        ent = df[cond1 & cond2].reset_index(drop=True)
        ent = ent.iloc[:top_n]
        return ent

    @staticmethod
    def trim_stop_words(s):
        if len(s) < 1:
            return s
        if len(s) == 1:
            s1 = [] if (s[0].text.lower() in STOP_WORDS) else s
        else:
            s1 = s[1:] if (s[0].text.lower() in STOP_WORDS) else s
            if len(s1) == 1:
                s1 = [] if (s[-1].text.lower() in STOP_WORDS) else s
            else:
                s1 = s1[:-1] if (s1[-1].text.lower() in STOP_WORDS) else s1
        return s1

    @staticmethod
    def trim_entities(s):
        if len(s) < 2:
            return s
        if s[-1].text == s.root.text:
            n = len(s) - 1
        else:
            n = len(s) - 2
        for i in range(n - 1, -1, -1):
            if not s[i].pos_ in ['NOUN', 'PROPN', 'ADJ', 'PUNCT']:
                s1 = s[i + 1:n + 1]
                break
        else:
            s1 = s[:n + 1]
        return s1

    @staticmethod
    def remove_email_and_punctuation(s):
        if len(s) < 1:
            return s
        s1 = s if not s.root.like_email else []
        if len(s1) > 0:
            s1 = s1[:-1] if (s1[-1].tag_ in ['POS']) else s1
            s1 = s1[1:] if (s1[0].tag_ in ['POS']) else s1
        if len(s1) > 0:
            s1 = s1[1:] if (s1[0].pos_ in ['PUNCT']) else s1
        return s1

    @staticmethod
    def merge_keywords_by_similarity(df, th=0.4):
        if len(df) > 1:
            m = calculate_cosine_similarity(df['entity'].tolist())
            most_sim = [[j for j in range(len(m)) if m[i, j] >= th] for i in range(len(m)) if i < 20]
            m[np.tril_indices(len(m), -1)] = 0
            m[m < th] = 0
            m[m >= th] = 1
            df['total_score'] = np.matmul(m, df['weight'].values.reshape(-1, 1))
            for idx, x in enumerate(most_sim):
                if len(x) <= 1:
                    continue
                a = df.loc[x, 'entity']
                df.loc[idx, 'entity'] = a.values[0]
            df.loc[df['total_score'] == 0.0, 'total_score'] = df.loc[df['total_score'] == 0.0, 'weight'] * 1.0
            df = df.drop_duplicates(subset=['entity']).reset_index(drop=True)
            df.sort_values(by=['total_score', 'start'], inplace=True, ascending=[False, True])
        else:
            df['total_score'] = df['weight'] if 'weight' in df.columns else 1
        return df

