from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_cosine_similarity(str_list_1, str_list_2=None):
    """Return a two-dimension array, similarities between str_list_1 and str_list_2
    If str_list_2 is None or empty, return similarities between str_list_1 and str_list_1
    """
    if not str_list_1:
        return np.array([[0.0]])
    if not str_list_2:
        vectors = get_vectors(str_list_1)
        return cosine_similarity(vectors[:len(str_list_1)])
    else:
        vectors = get_vectors(str_list_1 + str_list_2)
        len_x = len(str_list_1)
        return cosine_similarity(vectors[:len_x], Y=vectors[len_x:len(str_list_1 + str_list_2)])


def get_vectors(str_list, hash_threshold=500):
    text = [t.replace("&", "_") for t in str_list]
    if len(text) > hash_threshold:
        vectorizer = HashingVectorizer(n_features=10000, stop_words=None)
        try:
            m = vectorizer.fit_transform(text)
        except ValueError:
            text += ["random_string_a_p_w"]
            m = vectorizer.fit_transform(text)
    else:
        vectorizer = CountVectorizer(input=text, stop_words=None)
        try:
            vectorizer.fit(text)
        except ValueError:
            text += ["random_string_a_p_w"]
            vectorizer.fit(text)
        m = vectorizer.transform(text)
    return m.toarray()


if __name__ == '__main__':
    import time
    start = time.time()
    sim = calculate_cosine_similarity(['White House', '', 'S&P'],
                                      str_list_2=['White House', 'Donald Trump', 'S P', 'S&P'])
    end = time.time()
    print("time {}".format(end-start))
    print(sim)
