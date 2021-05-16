import re
import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
from cosine_similarity import get_cosine_sim
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

    @staticmethod
    def extract_entity(ner_document):
        ents = [ent for ent in ner_document.ents if ent.label_ in KeywordsExtract.allow_types]
        ents = [KeywordsExtract.trim_tags(ent, for_type='tag') for ent in ents]
        ents = [KeywordsExtract.remove_email_and_punctuation(ent) for ent in ents]
        ents = [ent for ent in ents if len(ent) > 0]
        ent_list = [[KeywordsExtract.lemma_last_word(ent), ent.start_char,
                     ent.end_char, ent.label_, ent.lemma_] for ent in ents]
        cols = ['entity', 'start', 'end', 'label', 'lemma']
        df = pd.DataFrame(ent_list, columns=cols)
        df = KeywordsExtract.filter_entities(df)
        if not df.empty:
            df['entity'] = df['entity'].str.strip()
        return df

    @staticmethod
    def extract_entity_weighted(title_txt, ner_document, title_weight=4):
        df = KeywordsExtract.extract_entity(ner_document)
        df['ent_type'] = 'entity'
        if not df.empty:
            df['weight'] = df['start'].apply(lambda x: title_weight if (x < len(title_txt)) else 1)
        else:
            df['weight'] = 1
        df_grp, df = KeywordsExtract.process_entity(df)
        return df_grp, df

    @staticmethod
    def extract_noun_chunk(ner_document, title_weight=2, title=""):
        chunks = [ch for ch in list(ner_document.noun_chunks) if (ch.root.ent_type_ == '')]
        chunks = [ch for ch in chunks if len(ch) > 0]
        chunks = [KeywordsExtract.trim_tags(ch) for ch in chunks]
        chunks = [KeywordsExtract.trim_stop_words(ch) for ch in chunks]
        chunks = [KeywordsExtract.remove_email_and_punctuation(ch) for ch in chunks]
        chunks = [KeywordsExtract.trim_entities(ch) for ch in chunks]
        chunks = [ch for ch in chunks if len(ch) > 0]
        chunks_list = [[KeywordsExtract.lemma_last_word(ch), ch.start_char,
                        ch.end_char, ch.label_, ch.lemma_] for ch in chunks]
        cols = ['entity', 'start', 'end', 'label', 'lemma']
        df = pd.DataFrame(chunks_list, columns=cols)
        df = KeywordsExtract.filter_entities(df)
        if not df.empty:
            df['entity'] = df['entity'].str.strip()
        df['ent_type'] = 'noun_chunk'
        if not df.empty:
            df['weight'] = df['start'].apply(lambda x: title_weight if (x < len(title)) else 1)
            df = df[(df['entity'].str.len() > 3) | df['entity'].str.isupper()]
        else:
            df['weight'] = 1
        df_grp, df = KeywordsExtract.process_entity(df)
        return df_grp, df

    @staticmethod
    def consolidate(list_df):
        if len(list_df) == 1:
            return list_df[0]
        df = list_df[0]
        for i in range(1, len(list_df)):
            df = df.append(list_df[i], ignore_index=True, sort=False)
        df = KeywordsExtract.filter_entities(df)
        df1 = df.copy()
        if df1.empty:
            return df1
        df1['weight'] = df1['weight'] * df1['weight']
        df1 = df1.groupby(['entity'])[['weight']].sum()
        df0 = df.drop_duplicates(subset=['entity'])
        df0 = df0.drop(['weight'], axis=1)
        df_merge = pd.merge(df0, df1, left_on='entity', right_index=True)
        df_merge.sort_values(by=['weight', 'start'], inplace=True, ascending=[False, True])
        df_merge = df_merge.reset_index(drop=True)
        return df_merge

    @staticmethod
    def process_entity(df):
        _months = KeywordsExtract.months
        if not df.empty:
            df = df[~df['entity'].isin(KeywordsExtract.remove_entities)]
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

    @staticmethod
    def filter_entities(df):
        if 'entity' in df.columns:
            # remove certain special characters
            df['entity'] = df['entity'].apply(lambda x: re.sub('\.|[\-\'\$\/\\\*\+\|\^\#\@\~\`]{2,}', '', str(x)))
            df = df[df['entity'].apply(lambda x: re.search('\(|\)|\[|\]|\"|\:|\{|\}|\^|\*|\;|\~|\|', str(x)) is None)]
            # remove unwanted words
            if not df.empty:
                df = df[df['entity'].apply(lambda x: x.lower() not in KeywordsExtract.remove_words)]
            # remove too long entities
            if not df.empty:
                df = df[df['entity'].apply(lambda x: len(str(x)) < 35)]
            # remove too short entities
            if not df.empty:
                df = df[df['entity'].apply(lambda x: len(str(x)) > 1)]
        return df

    @staticmethod
    def trim_tags(s, for_type='chunk', trim=['PDT', 'DT', 'IN', 'CC'], punctuation=[',', '\'']):
        if len(s) < 1:
            return s
        s1 = s
        if (for_type == 'chunk') and (re.search('|'.join(punctuation), s1.text) is not None):
            for i in range(len(s1) - 1, -1, -1):
                if s1[i].text in punctuation:
                    s1 = s1[i + 1:]
                    break
        if len(s1) > 0:
            s1 = s1[1:] if (s1[0].tag_ in trim) else s1
        if len(s1) > 1:
            s1 = s1[:-1] if (s1[-1].tag_ in trim) else s1
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
    def filter_entity(df, score_threshold=0.1, max_ents=50, top_n=5):
        if df.empty:
            return pd.DataFrame()
        elif len(df) > max_ents:
            df = df[:max_ents]
        cond1 = df['total_score'] / df['total_score'].sum() > score_threshold
        cond2 = df['total_score'] > 0
        ent = df[cond1 & cond2].reset_index(drop=True)
        ent = ent.iloc[:top_n]
        return ent

    @classmethod
    def extract_keywords(cls, title, txt, entity_weight=4, chunk_weight=2):
        title_txt = title + ". "
        full_content = title_txt + txt
        ner_document = cls.MODEL(full_content)
        df_ent_grp, df_ent = cls.extract_entity_weighted(title_txt, ner_document, title_weight=entity_weight)
        df_chunk_grp, df_chunk = cls.extract_noun_chunk(ner_document, title_weight=chunk_weight, title=title_txt)
        df_both = pd.DataFrame()
        if not df_ent_grp.empty:
            df_ent_grp = cls.merge_keywords_by_similarity(df_ent_grp)
            df_both = df_ent_grp.copy()
        if not df_chunk_grp.empty:
            df_chunk_grp = cls.merge_keywords_by_similarity(df_chunk_grp)
            df_both = df_both.append(df_chunk_grp)
        if not df_both.empty:
            df_both.sort_values(by=['total_score', 'start'], inplace=True, ascending=[False, True])
            df_both.drop_duplicates(subset=['lemma'], inplace=True)

        ent = cls.filter_entity(df_ent_grp, score_threshold=0.05, max_ents=30)
        chu = cls.filter_entity(df_chunk_grp, score_threshold=0.05, max_ents=30)
        if not chu.empty:
            ent = ent.append(chu, ignore_index=True, sort=False)
            ent.sort_values(by=['total_score', 'start'], inplace=True, ascending=[False, True])
            ent.drop_duplicates(subset=['entity'], inplace=True)
            ent.drop_duplicates(subset=['lemma'], inplace=True)
        if ent.empty:
            kwords_dict = []
        else:
            kwords_dict = [{'keyword': str(ent.loc[i, 'entity']), 'weight': int(ent.loc[i, 'total_score'])} for i in ent.index]
        return df_ent_grp, df_chunk_grp, kwords_dict

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
            m = get_cosine_sim(df['entity'].tolist())
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

