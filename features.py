import collections

from utils import sick_train_reader, sick_dev_reader, sick_test_reader, \
    str2tree, leaves



def keyword_overlap_feature(s1, s2, cioClient):
    sent1 = " ".join(leaves(s1))
    sent2 = " ".join(leaves(s2))

    s1Keywords = cioClient.extractKeywords(sent1)
    s2Keywords = cioClient.extractKeywords(sent2)

    return collections.Counter(set(s1Keywords) & set(s2Keywords))


features_mapping = {"keyword_overlap": keyword_overlap_feature}

def featurizer(reader=sick_train_reader, features_funcs=None, cioClient=None):
    """Map the data in reader to a list of features according to feature_function
    Valid feature_funcs return a dict of string : int key-value pairs.  """
    feats = []
    labels = []

    count = 0
    for label, t1, t2 in reader():
        if count % 10 == 0:
            print "Featurized %d examples" %(count)
        feat_dict = {} #Stores all features extracted using feature functions
        for feat in features_funcs:
            d = features_mapping[feat](t1, t2, cioClient)
            feat_dict.update(d)

        feats.append(feat_dict)
        labels.append(label)
        count += 1
    return (feats, labels)