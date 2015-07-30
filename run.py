""" Runs the model based on CIO fingerprints. """

import sklearn.metrics

from utils import sick_test_reader, sick_dev_reader, \
   sick_train_reader, leaves
from models import Baseline

def evaluateModel(model, reader):
   predictions = []
   goldLabels = []
   for label, t1, t2 in sick_test_reader:
      goldLabels.append(label)
      s1 = " ".join(leaves(t1))
      s2 = " ".join(leaves(t2))
      predict = model.predict(s1, s2)


if __name__ == "__main__":
   #for example in sick_test_reader():
   #    print example
   baseline = Baseline()
   evaluateModel(baseline, sick_test_reader)
