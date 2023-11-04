import torch
import torch.nn as nn
from utils import split_sentences, tokenize_sentences

class WordWindowClassifier(nn.Module):
    _allowed = ("window_size", "embed_size", "hidden_size", "vocab_size", "padding_idx", "freeze_embeddings")
    def __init__(self, window_size, embed_size, hidden_size, vocab_size, padding_idx, freeze_embeddings=False):
        super(WordWindowClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=padding_idx)
        self.window_size = window_size
        self.full_window_size = 2 * self.window_size + 1
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.full_window_size * self.embed_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.score = nn.Linear(self.hidden_size, 1)
        self.probabilities = nn.Sigmoid()
        self.freeze_embeddings = freeze_embeddings

    def set_property(self, **kwargs):
        for k, v in kwargs.iteritems():
            assert(k in self._allowed)
            setattr(self, k, v)

    def forward(self, batch_inputs):
        batch_size, _ = batch_inputs.size()
        # self.embeddings.weight.requires_grad = not self.freeze_embeddings
        batch_token_windows = batch_inputs.unfold(1, self.full_window_size, 1)
        sentence_length = batch_token_windows.size(1)
        assert batch_token_windows.size() == (batch_size, sentence_length, self.full_window_size)
        batch_embedding_windows = self.embeddings(batch_token_windows).view(batch_size, sentence_length, -1)
        hidden_input = self.relu(self.linear(batch_embedding_windows))
        batch_scores = self.score(hidden_input)
        batch_outputs = self.relu(batch_scores)
        return batch_outputs.squeeze(2)

    def predict_raw(self, raw_sentences, word2index):
        pad_tokens = [word2index['<pad>']] * self.window_size
        padded_inputs = [torch.LongTensor(pad_tokens + tokenized_sentence + pad_tokens) for tokenized_sentence in tokenize_sentences(split_sentences(raw_sentences), word2index=word2index)]
        inputs = torch.nn.utils.rnn.pad_sequence(padded_inputs, batch_first=True, padding_value=word2index['<pad>'])
        return self.forward(inputs)
