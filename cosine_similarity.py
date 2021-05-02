from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_cosine_sim(str_list_1, str_list_2=None):
    """Return a two-dimension array, similarities between str_list_1 and str_list_2
    If str_list_2 is None or empty, return similarities between str_list_1 and str_list_1
    """
    if not str_list_1:
        return np.array([[0.0]])
    if not str_list_2:
        vectors = [t for t in get_vectors(str_list_1)]
        return cosine_similarity(vectors[:len(str_list_1)])
    else:
        vectors = [t for t in get_vectors(str_list_1 + str_list_2)]
        len_x = len(str_list_1)
        return cosine_similarity(vectors[:len_x], Y=vectors[len_x:len(str_list_1 + str_list_2)])


def get_vectors(str_list):
    str_list = [t for t in str_list if t]
    text = [t.replace("&", "_") for t in str_list]
    try:
        vectorizer = CountVectorizer(stop_words=None, input=text)
        vectorizer.fit(text)
    except ValueError:
        text += ["random_string_a_p_w"]
        vectorizer = CountVectorizer(stop_words=None, input=text)
        vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


if __name__ == '__main__':
    sim = get_cosine_sim(['Eric trump'],
                         str_list_2=['White House', 'Donald Trump', ''])
    print(type(sim), sim)
