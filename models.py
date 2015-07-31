""" Models using various Cio utilities """

import cortipy

from features import featurizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import sick_train_reader, sick_dev_reader, sick_test_reader


class Model(object):
    """Base model class
    """
    def __init__(self, metric, featureFuncs):
        self.client = cortipy.CorticalClient(useCache=True) # Or False?
        self.metric = metric # Name of desired metric to use for comparison
        self.featureFuncs = featureFuncs

    def train(self):
        raise NotImplementedError

    def predict(self, s1, s2):
        """A single sentence pair."""
        raise NotImplementedError

    def predictAll(self, feats):
        """Predict labels for all samples in a feature list."""



class Baseline(Model):
    """ Does a simple comparison of two texts using fingerprint comparison of
    unpreprocessed text.
    """
    def __init__(self, metric, featureFuncs):
        super(Baseline, self).__init__(metric, featureFuncs)

    def train(self):
        pass

    def predict(self, s1, s2):
        """ Returns the predicted label
        """

        # Do preprocessing with keyword extraction later
        # s1Keywords = self.client.extractKeywords(s1)
        # s2Keywords = self.client.extractKeywords(s2)

        try: # Error for sentence: A player is throwing the ball
            bitmapS1 = self.client.getTextBitmap(s1)
            bitmapS2 = self.client.getTextBitmap(s2)
        except cortipy.exceptions.UnsuccessfulEncodingError:
            return "NEUTRAL"

        score = self.client.compare(bitmapS2["fingerprint"]["positions"], bitmapS1["fingerprint"]["positions"])
        try:
            desiredScore = score[self.metric]
        except KeyError:
            print "Error metric %s does not exist" %(self.metric)

        # Foolish mechanism:  0-0.33 -> contradiction
        #                     0.34-0.66 -> neutral
        #                     0.67-1.0 -> entailment
        if desiredScore <= 0.33:
            return  "CONTRADICTION"
        elif desiredScore <= 0.66:
            return "NEUTRAL"
        else:
            return "ENTAILMENT"

    def predictAll(self, feats):
        pass



class Keyword(Model):
    """ Does a simple comparison of two texts using fingerprint comparison of
    unpreprocessed text.
    """
    def __init__(self, metric, featureFuncs):
        super(Keyword, self).__init__(metric, featureFuncs)

    def train(self):
        pass

    def predict(self, s1, s2):
        """ Returns the predicted label
        """

        # Do preprocessing with keyword extraction later
        s1Keywords = self.client.extractKeywords(s1)
        s2Keywords = self.client.extractKeywords(s2)

        reducedS1 = " ".join(s1Keywords)
        reducedS2 = " ".join(s2Keywords)

        try: # Error for sentence: A player is throwing the ball
            bitmapS1 = self.client.getTextBitmap(reducedS1)
            bitmapS2 = self.client.getTextBitmap(reducedS2)
        except cortipy.exceptions.UnsuccessfulEncodingError:
            return "NEUTRAL"

        score = self.client.compare(bitmapS2["fingerprint"]["positions"], bitmapS1["fingerprint"]["positions"])
        try:
            desiredScore = score[self.metric]
        except KeyError:
            print "Error metric %s does not exist" %(self.metric)

        # Foolish mechanism:  0-0.33 -> contradiction
        #                     0.34-0.66 -> neutral
        #                     0.67-1.0 -> entailment
        if desiredScore <= 0.33:
            return  "CONTRADICTION"
        elif desiredScore <= 0.66:
            return "NEUTRAL"
        else:
            return "ENTAILMENT"

    def predictAll(self, feats):
        pass


class NaiveBayes(Model):
    """ Naive bayes model
    """
    def __init__(self, metric, featureFuncs):
        self.classifier = MultinomialNB()
        self.dictVectorizer = DictVectorizer()
        super(NaiveBayes, self).__init__(metric, featureFuncs)

    def train(self):
        feats, labels = featurizer(sick_train_reader, self.featureFuncs,
                                   self.client)
        featMat = self.dictVectorizer.fit_transform(feats)
        self.classifier.fit(featMat, labels)

    def predict(self, s1, s2):
        pass

    def predictAll(self, reader):
        """Prediction function for all features from a certain reader."""
        feats, labels = featurizer(reader, self.featureFuncs,
                                   self.client)
        featMat = self.dictVectorizer.transform(feats)
        return self.classifier.predict(featMat)

