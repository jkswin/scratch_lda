import unittest
import pandas as pd
import os
import json

from lda import LDA

"""
Exemplar Case Study: Similarity comparison of reviews of each of 19 McDonald's stores in the US.
Speculative use as a means of classifying opaque "store types" from a given store's reviews. 

"""

class LDATest(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        #init the data
        self.df = pd.read_json("lda/data/mcdonalds_nlp.json")
        self.vocab = list(self.df["tf_limited"][0].keys()) 
        self.document_term_dict = {name:list(bow.values()) for name, bow in zip(self.df["short_address"], self.df["tf_limited"])}
        self.document_term_matrix = list(self.document_term_dict.values())
        self.names = self.df["short_address"]

        #init the model
        #let's make an assumption that reviews tend to focus on 3 topics
        # cleanliness, food, service
        self.model = LDA(n_topics=3, 
                         n_iters=1000, 
                         random_state=42,
                         alpha=0.01, #we'll lower the def alpha to say that there arent many topics per document
                         beta=0.2 #and increase the beta to assume that lots of words a related to each topic
                         )
        
        self.path = "lda/models/test_lda_model.pkl"

    @unittest.SkipTest
    def test_fit(self):
        # note that this takes ~40 mins on my machine with the settings above
        # hence why I have saved a pkl and skipped this test 
        self.model.fit(self.document_term_matrix, self.vocab)
        self.model.print_topics()
        self.model.save_pretrained(path=self.path)

    def test_vis(self):
        self.model = LDA.from_pretrained(path=self.path)
        self.model.topic_term_heatmap()
        self.model.doc_topic_heatmap(yticklabels=self.names)
        


if __name__ == "__main__":
    unittest.main()