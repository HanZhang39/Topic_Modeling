import random
import logging
import numpy as np


logger = logging.getLogger('lda')

class LDA:
    """
    LDA topic modeling with gibbs sampling implemented according to


    k topic num
    """
    def __init__(self, corpus, id_word_dict, alpha=0.1, beta=0.1, k=10, num_iter=20):

        """
        z   which topic each word assigned to
        k   number of topic

        alpha, beta priori

        n_w[i][w] number of word w assigned to topic i
        n[i]      total number of word from topic i
        n_d[j][i] number of word from topic i in document j

        """
        self.z = [sum([[(idx, random.randrange(k)) for _ in range(cnt)]
                       for idx, cnt in doc.items()], [] ) for doc in corpus]
        """
        n_w number of word w assign to topic
        n   number of word assign to topic

        """
        self.z = [sum([[(idx, random.randrange(k)) for _ in range(cnt)]
                       for idx, cnt in doc.items()], [] ) for doc in corpus]
        self.corpus = corpus
        self.alpha = alpha
        self.beta = beta
        self.vocab_size = len(id_word_dict)
        self.k = k

        self.n_w = np.zeros((k, len(id_word_dict)))
        self.n_d = np.zeros((len(corpus), k))
        self.n = np.zeros(k)

        for idx, doc in enumerate(self.z):
            for w, topic in doc:
                self.n_w[topic][w] += 1
                self.n_d[idx][topic] += 1
                self.n[topic] += 1

        for i in range(num_iter):
            logger.debug(f"Gibbs sampling.. iter:{i}")
            self.__gibbs__()

    def __gibbs__(self):
        """cancel assignment"""
        def probability(topic, word, idx):
            return ((self.n_w[topic][w] + self.beta) * (self.n_d[idx][topic] + self.alpha)) / ((self.n[topic] + self.vocab_size * self.beta) * (len(self.corpus[idx]) - 1 + self.k * self.alpha))


        for idx, doc in enumerate(self.z):
            for idx_w, (w, topic) in enumerate(doc):

                self.n_w[topic][w] -= 1
                self.n_d[idx][topic] -= 1
                self.n[topic] -= 1

                prob = [probability(t, w, idx) for t in range(self.k)]
                total = sum(prob)
                prob = [p / total for p in prob]
                topic = np.random.choice(self.k, p=prob)

                self.n_w[topic][w] += 1
                self.n_d[idx][topic] += 1
                self.n[topic] += 1
                doc[idx_w] = (w, topic)


    def show_topic(self):
        topics = [[(cnt, w) for w, cnt in enumerate(t)] for t in self.n_w]
        topics = [sorted(t, reverse=True) for t in topics]
        return topics

    def show_document_topic(self):
        pass
