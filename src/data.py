import os
import pandas as pd
import random
import torch
from utils import pickle_obj, unpickle_obj


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, csv_filepath, encoding='utf-8', vocab_filepath='vocab.pkl'):
        # Can be 'train', 'test', or 'validate'. Default is 'train'
        self.mode = 'train'

        self.label2tag = [
            'O',        # Untagged
            'B-geo',    # Geographical Entity (Beginning)
            'I-geo',    # Geographical Entity (Inside)
            'B-org',    # Organization (Beginning)
            'I-org',    # Organization (Inside)
            'B-per',    # Person (Beginning)
            'I-per',    # Person (Inside)
            'B-gpe',    # Geopolitical Entity (Beginning)
            'I-gpe',    # Geopolitical Entity (Inside)
            'B-tim',    # Time indicator (Beginning)
            'I-tim',    # Time indicator (Inside)
            'B-art',    # Artifact (Beginning)
            'I-art',    # Artifact (Inside)
            'B-eve',    # Event (Beginning)
            'I-eve',    # Event (Inside)
            'B-nat',    # Natural Phenomenon (Beginning)
            'I-nat',    # Natural Phenomenon (Inside)
        ]
        self.tag2label = {tag: index for index, tag in enumerate(self.label2tag)}

        self.csv_filepath = csv_filepath

        # Read CSV file as all string columns and truncate to only necessary columns
        df = pd.read_csv(csv_filepath, encoding=encoding, dtype=str, keep_default_na=False)
        df = df[['sentence_marker', 'word', 'tag']]
        
        # Store vocab filepath
        self.vocab_filepath = vocab_filepath
        
        # Tokenize words as indices
        if os.path.isfile(self.vocab_filepath):
            # Attempt to read word tokens from pickle file, if it exists
            self.index2word = unpickle_obj(self.vocab_filepath)
        else:
            # Otherwise create from scratch
            self.index2word = ['<pad>', '<unk>'] + df['word'].unique().tolist()
            pickle_obj(self.index2word, self.vocab_filepath)

        self.word2index = {word: index for index, word in enumerate(self.index2word)}
        
        assert df.iloc[0]['sentence_marker']

        # Create the base 'sentences' dataset
        self.all_sentences_count = -1
        self.all_sentences = []
        for index, row in df.iterrows():
            if row['sentence_marker']:
                self.all_sentences.append([])
                self.all_sentences_count += 1
            self.all_sentences[self.all_sentences_count].append({
                'word': row['word'], 
                'input': self.word2index[row['word']], 
                'label': self.tag2label.get(row['tag'], self.tag2label['O'])
            })
        
        # Create train, test and validate datasets
        random.shuffle(self.all_sentences)
        self.train_sentences_count = int(self.all_sentences_count * 0.7)
        new_sentence_count = self.all_sentences_count - self.train_sentences_count
        self.test_sentences_count = int(new_sentence_count * 0.5)
        test_start_index = self.train_sentences_count
        validate_start_index = self.train_sentences_count + self.test_sentences_count
        self.train_sentences = self.all_sentences[0:test_start_index]
        self.test_sentences = self.all_sentences[test_start_index:validate_start_index]
        self.validate_sentences = self.all_sentences[validate_start_index:]
        self.validate_sentences_count = len(self.validate_sentences)

    def __len__(self):
        if self.mode == 'train':
            return self.train_sentences_count
        elif self.mode == 'test':
            return self.test_sentences_count
        elif self.mode == 'validate':
            return self.validate_sentences_count
        else:
            return self.all_sentences_count
    
    def __getitem__(self, index):
        if self.mode == 'train':
            return self.train_sentences[index]
        elif self.mode == 'test':
            return self.test_sentences[index]
        elif self.mode == 'validate':
            return self.validate_sentences[index]
        else:
            return self.all_sentences[index]
    
    def set_mode(self, mode):
        allowed_modes = ['train', 'test', 'validate', 'all']
        if (not mode in allowed_modes):
            raise ValueError(f"Invalid mode '{mode}' specified. Allowed modes are [{', '.join(allowed_modes)}]")
        self.mode = mode

