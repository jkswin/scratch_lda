import numpy as np
from tqdm import tqdm, trange
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

class LDA:

    """
    Implementation of Latent Dirichlet Allocation based on Blei et al. 2003
    It is a generative probabilistic model used for topic modeling, 
    which is an unsupervised problem within NLP.
    The assumption is that a document is a mixture of topics, 
    and each topic is a distribution over words.

    Latent = Hidden; the topics are not known 

    Dirichlet = Probability Distributions to model uncertainty 
                in topic proportions for documents
                and in word proportions within topics

    Allocation = Words assigned to Topics within the documents

    alpha parameter - document-topic density: larger alpha = more assumed topics per document
    beta parameter - topic-word density: larger beta = more words assumed per topic

    Gibbs Sampling is used to estimate the posterior distribution of latent variables 
    (topics) given observed data (the document-term matrix).
    p(x,y) is difficult to determine, 
    but p(x|y) and p(y|x) are easy because they are single sample distributions.
    GS freezes one of the variables x,y while sampling from the other.


    Overview:
    - LDA Summary https://www.youtube.com/watch?v=1_jq_gWFUuQ
    - Dirichlet Distribution Summary https://www.youtube.com/watch?v=nfBNOWv1pgE&t=24s
    - Gibbs Sampling (V good vid) https://www.youtube.com/watch?v=7LB1VHp4tLE 
    - 

    """
    def __init__(self, n_topics, alpha=0.1, beta=0.1, n_iters=10, random_state=None) -> None:
        self.n_topics = n_topics
        self.alpha = alpha # start off as symmetric for topics
        self.beta = beta # start off as symmetric for topics
        self.n_iters = n_iters

        if random_state is not None:
            np.random.seed(random_state)


    def fit(self, document_term_matrix, vocab):

        self.document_term_matrix: dict = document_term_matrix
        self.vocab: list[int] = vocab
        self.tok2idx: dict = {t:i for i,t in enumerate(self.vocab)}
        self.idx2tok: dict = {i:t for t,i in self.tok2idx.items()}
        self.vocab_size: int = len(self.vocab)
        self.beta=np.ones(self.vocab_size) * self.beta

        self._initialise_params()

        for _ in trange(self.n_iters):
            self._gibbs_sampling()

    def _initialise_params(self):
        """
        Randomly assign a topic to each document.
        Establish initial distributions of topics based on counts of each word within documents.
        Including:
            - Counts of each topic within each doc
            - Counts of each word within each topic 
        """

        # keep track of the topic associated with each document. (n_docs x 1)
        self.doc_topics = np.random.choice(self.n_topics, size=len(self.document_term_matrix))
        # keep track of the number of topics associated with each doc (n_docs x n_topics)
        self.doc_topic_counts = np.zeros((len(self.document_term_matrix), self.n_topics), dtype=int)
        # keep track of the number of terms within each topic (n_topics x vocab_size)
        self.topic_term_counts = np.zeros((self.n_topics, self.vocab_size), dtype=int)

        # iterate over documents, words 
        for doc_idx, doc in enumerate(self.document_term_matrix):
            for word_idx, count in enumerate(doc):
                # for each word, it's current topic is the overarching document topic
                topic = self.doc_topics[doc_idx]
                word_id = word_idx
                # to the topic counts for each document, add the count of this word
                self.doc_topic_counts[doc_idx, topic] += count
                # to the number of terms associated with this topic, add the count of this word
                self.topic_term_counts[topic, word_id] += count



    def _gibbs_sampling(self):
        """
        TLDR: Estimate distributions of hidden topics based on observed data: the corpus.
        """

        for doc_idx, doc in enumerate(self.document_term_matrix):
            for word_idx, count in enumerate(doc):
                topic = self.doc_topics[doc_idx]
                word_id = word_idx
                self.doc_topic_counts[doc_idx, topic] -= count
                self.topic_term_counts[topic, word_id] -= count
                # the heavy lifting
                p_topic = (self.doc_topic_counts[doc_idx] + self.alpha) * \
                        (self.topic_term_counts[:, word_id] + self.beta[word_id]) / \
                        (self.topic_term_counts.sum(axis=1) + self.beta.sum())
                cumulative_p_topic = np.cumsum(p_topic)
                sample = np.random.random() * cumulative_p_topic[-1]
                new_topic = np.searchsorted(cumulative_p_topic, sample) - 1
                self.doc_topic_counts[doc_idx, new_topic] += count
                self.topic_term_counts[new_topic, word_id] += count
                self.doc_topics[doc_idx] = new_topic


    def print_topics(self, num_words=10):
        """
        Print words associated with each topic.
        """
        for topic_idx in range(self.n_topics):
            topic_words = [word for word_idx, word in sorted([(word_id, word) for word, word_id in self.tok2idx.items()],
                                                     key=lambda x: self.topic_term_counts[topic_idx, x[0]], reverse=True)[:num_words]]
            print(f"Topic {topic_idx}: {', '.join(topic_words)}")

    
    def topic_term_heatmap(self, percentile=99):
        """
        Display a heatmap of topics-words topics from the nth percentile of words for each topic.
        """

        prom_index = []

        # make a list of words that are prominent in topics
        for topic_id in range(self.n_topics):
            perc = np.percentile(self.topic_term_counts[topic_id], percentile)
            for word_id in range(self.vocab_size):
                if self.topic_term_counts[topic_id][word_id] > perc:
                    prom_index.append(word_id)

        prom_index = sorted(list(set(prom_index)))
        prom_words = [self.idx2tok[i] for i in prom_index]
        # make topic-term matrix using only these indexes
        prom_matrix = self.topic_term_counts[:, prom_index]
        
        fig, ax = plt.subplots(figsize=(15,8)) 
        sns.heatmap(prom_matrix, xticklabels=prom_words, ax=ax, center=0)
        plt.title("Learned Word-Topic Associations")
        plt.yticks(rotation=90)
        plt.ylabel("Topic")
        plt.show()

    def doc_topic_heatmap(self, **kwargs):
        if "yticklabels" in kwargs.keys():
            kwargs["yticklabels"] = [ '\n'.join(wrap(l, 20)) for l in kwargs["yticklabels"] ]

        fig, ax = plt.subplots(figsize=(15,8))
        sns.heatmap(self.doc_topic_counts, **kwargs, ax=ax, center=0)
        plt.title("Document-Topic Relations")
        plt.show()


    def save_pretrained(self, path:str):
        """
        Pickle the model.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
    

    @classmethod
    def from_pretrained(cls, path:str):
        """
        Load model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    


