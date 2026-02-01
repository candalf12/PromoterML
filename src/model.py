import torch
import torch.nn as nn
import torch.nn.functional as F

class DNA_Classifier(nn.Module):
    # num_motifs = 16 Number of patterns for the model to learn
    # motif_len = 10 How long each pattern is 
    def __init__(self, seq_length, num_motifs=16, motif_len=10):
        super(DNA_Classifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels= num_motifs, kernel_size=motif_len)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)

        conv_out_size = (seq_length - motif_len +1) // 3
        self.fc_input_dim = num_motifs * conv_out_size

        self.fc1 = nn.Linear(self.fc_input_dim,1)  
    def forward(self,x):

        x = F.relu(self.conv1(x)) # first convulation then the activation function (to remove - output)
        x = self.pool(x) # shrink the data

        x = x.view(x.size(0),-1) # Flatten the dataset into a 2D Matrix

        x = torch.sigmoid(self.fc1(x))
        return x
      