import os
import pickle
import re
import sys
import torch

def split_sentences(sentences):
    return [re.split('\W+', sentence) for sentence in sentences]

def tokenize_sentences(sentences, word2index):
    result = [None] * len(sentences)
    for index, sentence in enumerate(sentences):
        result[index] = []
        for word in sentence:
            result[index].append(word2index.get(word, word2index['<unk>']))
    return result

def relative_path(filepath):
    caller__file__ = sys._getframe(1).f_globals['__file__']
    print(f"caller__file__: {caller__file__}")
    caller_dirname = os.path.dirname(caller__file__)
    print(f"caller_dirname: {caller_dirname}")
    print(f"filepath: {filepath}")
    print(f"os.path.join test: {os.path.join(caller_dirname, filepath)}")
    return os.path.join(caller_dirname, filepath)

def pickle_obj(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def unpickle_obj(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def save_model_to_path(model, path):
    torch.save(model.state_dict(), path)

# TODO: make load_model_from_path