import torch

def one_hot_encoding(seq_string):
    mapping = { 'A' :0, 'C': 1, 'G' : 2, 'T' : 3}
    seq_len = len(seq_string)
    tensor = torch.zeros(4, seq_len) # creating a matrix with 4 rows

    for i, char in enumerate(seq_string):
        if char in mapping:
            row_index = mapping[char]
            tensor[row_index, i] = 1.0
    return tensor

def collate_fn(batch):
    sequences, labels = zip(*batch)
    encoded_seqs = [one_hot_encoding(s) for s in sequences] # processing the sequence
    X = torch.stack(encoded_seqs) #transformation from 32 matrices to 1 big matrix
    y = torch.tensor(labels, dtype=torch.float32) # tensor of floats for loss calculation

    return X, y
