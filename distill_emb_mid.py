import torch
import torch.nn as nn


class DistillEmbMid(nn.Module):
    
    def __init__(self, char2int, output_size, pad_char, char_emb_size=64, kernel=5, dropout=0.1):
        super(DistillEmbMid, self).__init__()

        self.n_inputs = 15 # number of characters
        self.output_size = output_size
        self.char2int = char2int
        self.pad_char = pad_char

        self.embedding = nn.Embedding(len(char2int), char_emb_size)
        self.conv1 = nn.Conv1d(self.n_inputs, 64, kernel, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel, stride=1)
        self.conv3 = nn.Conv1d(128, 192, kernel, stride=1)
        self.conv4 = nn.Conv1d(192, 256, 3, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, output_size)

        self.activation = nn.Mish()

        self.norm0 = nn.LayerNorm([self.n_inputs, char_emb_size])
        self.norm1 = nn.LayerNorm([64, 30])
        self.norm2 = nn.LayerNorm([128, 13])
        self.norm3 = nn.LayerNorm([192, 4])
        self.norm4 = nn.LayerNorm(512)
        self.output_norm = nn.LayerNorm(512)

        self.dropout = nn.Dropout(dropout)
    
    def embed(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.norm0(x)

        x = self.conv1(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm1(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm2(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm3(x)

        x = self.conv4(x)
        x = x.view(x.shape[0], -1)

        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm4(x)
        x = self.fc1(x)

        return x
        
    def forward(self, x):
        x = self.embed(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_norm(x)

        x = self.output_layer(x)
        return x

    def init_model(self, model_name=None):

        if model_name != None:
            checkpoint = torch.load(model_name)
            self.load_state_dict(checkpoint['model_state_dict'])

if __name__ == "__main__":
    m = DistillEmbMid([0]*400, 300, 0, 64, 5)
    import numpy as np
    x = np.random.randint(0, 400, (10, 15))
    x = torch.tensor(x, dtype=torch.int64)
    y = m(x)
    print(y.shape)