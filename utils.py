import csv
import os
import re
import sys


WORD_RE = re.compile(r"([^ \(\)]+)", re.UNICODE)

labels = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]

def str2tree(s):
    """Turns labeled bracketing s into a tree structure (tuple of tuples)"""
    s = WORD_RE.sub(r'"\1",', s)
    s = s.replace(")", "),").strip(",")
    s = s.strip(",")
    return eval(s)

def leaves(t):
    """Returns all of the words (terminal nodes) in tree t"""
    words = []
    for x in t:
        if isinstance(x, str):
            words.append(x)
        else:
            words += leaves(x)
    return words

data_dir = ''

def sick_reader(src_filename):
    for example in csv.reader(file(src_filename), delimiter="\t"):
        label, t1, t2 = example[:3]
        if not label.startswith('%'): # Some files use leading % for comments.
            yield (label, str2tree(t1), str2tree(t2))

# Readers for processing SICK datasets
def sick_train_reader():
    return sick_reader(src_filename=data_dir+"SICK_train_parsed.txt")

def sick_dev_reader():
    return sick_reader(src_filename=data_dir+"SICK_dev_parsed.txt")

def sick_test_reader():
    return sick_reader(src_filename=data_dir+"SICK_test_parsed.txt")
