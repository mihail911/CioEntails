""" Models using various Cio utilities """

import cortipy

class Baseline(object):
    """ Does a simple comparison of two texts using fingerprint comparison of
    unpreprocessed text.
    """
    def __init__(self):
        self.client = cortipy.CorticalClient(useCache=True) # Or False?

    def train(self):
        pass

    def predict(self, s1, s2):
        keywords = self.client.extractKeywords("dogs are the coolest animals and friends that have ever existed")
        print "Keywords: ", keywords




class NaiveBayes(object):
    """ Naive bayes model
    """

    def __init__(self):
        pass
