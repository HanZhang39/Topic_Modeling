import random
import logging
import numpy as np
import sys

logger = logging.getLogger('btm')
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


def construct_sentence_biterm(sentence, window=20):
    return sum([[tuple(sorted((w, sentence[i + j]))) for j in range(1, window) if i + j < len(sentence)] for i, w in enumerate(sentence)], [])

def construct_biterm(corpus, window=20):
    return [construct_sentence_biterm(doc, window) for doc in corpus]

class BTM:
    """
    LDA topic modeling with gibbs sampling implemented according to


    k topic num
    """
    def __init__(self, id_corpus, id_word_dict, window=5, alpha=0.001, beta=0.01, k=1000, num_iter=20):

        """
        z   which topic each word assigned to
        k   number of topic
        window context window size for generating biterms


        alpha, beta priori

        n_w[i][w] number of word w assigned to topic i
        n[i]      total number of word from topic i
        n_d[j][i] number of word from topic i in document j

        """
        biterm_corpus = construct_biterm(id_corpus, window)

        self.z = [[(biterm, random.randrange(k))
                   for biterm in doc] for doc in biterm_corpus]
        """
        n_w number of word w assign to topic
        n   number of word assign to topic

        """
        self.corpus = biterm_corpus
        self.alpha = alpha
        self.beta = beta
        self.vocab_size = len(id_word_dict)
        self.k = k
        self.id_word_dict = id_word_dict

        # number of word i assigned to topic j
        self.n_w = np.zeros((k, len(id_word_dict)))
        # number of biterm i assigned to topic j
        self.n_z = np.zeros((k, len(id_word_dict), len(id_word_dict)))
        # number of word assigned to topic j
        self.n_d = np.zeros(k)
        self.n = np.zeros(k)

        for idx, doc in enumerate(self.z):
            for ((w_0, w_1), topic) in doc:
                self.n_w[topic][w_1] += 1
                self.n_w[topic][w_0] += 1
                self.n_z[topic][w_0][w_1] += 1
                self.n_d[topic] += 2
                self.n[topic] += 1

        for i in range(num_iter):
            logger.debug(f"Gibbs sampling.. iter:{i}")
            sys.stdout.flush()
            self.__gibbs__()

    def __gibbs__(self):
        """cancel assignment"""
        def probability(topic, w_0, w_1, idx):
            # What is M in original paper
            return ((self.n_z[topic][w_0][w_1] + self.alpha) *
                    (self.n_w[topic][w_0] + self.beta) *
                    (self.n_w[topic][w_1] + self.beta) /
                    (self.n_d[topic] + self.vocab_size * self.beta) ** 2
                   )

        for idx, doc in enumerate(self.z):
            for (idx_w, ((w_0, w_1), topic)) in enumerate(doc):
                self.n_w[topic][w_1] -= 1
                self.n_w[topic][w_0] -= 1
                self.n_z[topic][w_0][w_1] -= 1
                self.n_d[topic] -= 2
                self.n[topic] -= 1

                prob = [probability(t, w_0, w_1, idx) for t in range(self.k)]
                total = sum(prob)
                prob = [p / total for p in prob]
                topic = np.random.choice(self.k, p=prob)

                self.n_w[topic][w_1] += 1
                self.n_w[topic][w_0] += 1
                self.n_z[topic][w_0][w_1] += 1
                self.n_d[topic] += 2
                self.n[topic] += 1

                doc[idx_w] = ((w_0, w_1), topic)


    def show_topic(self, top_n=20):
        def phi(word, topic):
            return (self.n_w[topic][word] + self.beta) / (self.n_d[topic] + self.vocab_size * self.beta)


        topics = [[(phi(w_id, t_id), w_id) for w_id in range(self.vocab_size)]
                  for t_id in range(self.k)]
        topics = [sorted(t, reverse=True) for t in topics]
        return [[(weight, self.id_word_dict[word]) for weight, word in t[:20]] for t in topics]

    def show_document_topic(self):
        pass

