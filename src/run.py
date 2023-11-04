from data import NERDataset
from functools import partial
from model import WordWindowClassifier
import os
import torch
from torch.optim import Adam
import time
from train import train

def custom_collate_fn(batch, word2index, tag2label, window_size=3):
    batch_len = len(batch)
    batch_inputs = [None] * batch_len
    batch_labels = [None] * batch_len
    batch_label_lengths = [None] * batch_len
    pad_list = [word2index['<pad>']] * window_size
    for index, row in enumerate(batch):
        batch_inputs[index] = torch.LongTensor(pad_list + [word['input'] for word in row] + pad_list)
        batch_labels[index] = torch.LongTensor([word['label'] for word in row])
        batch_label_lengths[index] = len(batch_labels[index])
    batch_inputs = torch.nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True, padding_value=word2index['<pad>'])
    batch_labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=tag2label['O'])
    batch_label_lengths = torch.LongTensor(batch_label_lengths)
    return batch_inputs, batch_labels, batch_label_lengths

if __name__ == "__main__":
    vocab_filepath = input('Enter vocabulary filepath (vocab.pkl): ') or "vocab.pkl"
    window_size = int(input('Enter window size: ') or "3")
    num_epochs = int(input('Enter number of epochs: ') or "5")
    batch_size = int(input('Enter batch size: ') or "32")
    learning_rate = float(input('Enter initial learning rate: ') or "0.01")
    dataset = NERDataset('data/ner_dataset.csv', encoding='ISO-8859-1', vocab_filepath=vocab_filepath)
    collate_fn = partial(custom_collate_fn, word2index=dataset.word2index, tag2label=dataset.tag2label, window_size=window_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    print("Dataset configured:")
    print(f"    Train dataset length: {len(dataset.train_sentences)}")
    print(f"    Test dataset length: {len(dataset.test_sentences)}")
    print(f"    Validate dataset length: {len(dataset.validate_sentences)}")

    vocab_size = len(dataset.index2word)
    padding_idx = dataset.word2index['<pad>']
    model = WordWindowClassifier(window_size=window_size, embed_size=50, hidden_size=30, vocab_size=vocab_size, padding_idx=padding_idx)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train(model, loader, optimizer, num_epochs)

    print('Model training complete:')

    for name, parameters in model.named_parameters():
        print(f"{name}: {parameters}\n")
    
    input_sentences = ['This is the great and wonderful Eiffel Tower', 'I went to Washington DC for pumpkin pie', 'We saw the statue of George Washington', 'The statue of David is tall', 'My hat was placed on top of the Mona Lisa', 'Samsung makes red and blue packaging for their customers']
    print(model.predict_raw(input_sentences, word2index=dataset.word2index))

"""
Ideal training parameters so far:

Enter window size: 3
Enter number of epochs: 30
Enter batch size: 2
Enter initial learning rate: 0.001

Try to get inference to work now.
"""