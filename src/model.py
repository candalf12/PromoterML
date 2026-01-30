import torch
import torch.nn as nn
import torch.nn.functional as F

class DNA_Classifier(nn.Module):
    num_motifs = 16 
    motif_len = 10
    def __init__(self, seq_length, num_motifs, motif_len):
        super(DNA_Classifier, self).__init__()
      