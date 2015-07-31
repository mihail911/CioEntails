""" Runs the model based on CIO fingerprints. """

import argparse
import time

from features import featurizer
from sklearn.metrics import accuracy_score
from models import Baseline, Keyword
from utils import sick_test_reader, sick_dev_reader, \
   sick_train_reader, leaves



def evaluateModel(model, reader):
   if reader == sick_dev_reader:
      dataSet = "dev"
   elif reader == sick_test_reader:
      dataSet = "test"
   else:
      dataSet = "train"

   predictions = []
   goldLabels = []
   count = 0
   for label, t1, t2 in reader():
      if count % 10 == 0:
         print "Processed %d examples" %(count)
      goldLabels.append(label)
      s1 = " ".join(leaves(t1))
      s2 = " ".join(leaves(t2))
      modelPredict = model.predict(s1, s2)
      predictions.append(modelPredict)
      count += 1

   accuracy = accuracy_score(predictions, goldLabels)
   print "Accuracy on SICK %s set: %f" %(dataSet, accuracy)


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="arguments for CioEntails system")
   parser.add_argument("--model",
                       type=str,
                       default="baseline",
                       help="name of model to use for system")
   args = parser.parse_args()

   if args.model == "baseline":
      model = Baseline("cosineSimilarity", ["keyword_overlap"])
   elif args.model == "keyword":
      model = Keyword("cosineSimilarity", ["keyword_overlap"])

   start = time.time()
   evaluateModel(model, sick_dev_reader)
   print "Evaluation done in %f seconds" %(time.time() - start)
