import torch
import torch.nn as nn
import numpy as np
import lib

class DistillEmb(nn.Module):
    
    def __init__(self, char2int, output_size, pad_char, char_emb_size=64, kernel=5, dropout=0.1, model_size='small'):
        super(DistillEmb, self).__init__()

        self.n_inputs = 13 # number of characters
        self.output_size = output_size
        self.char2int = char2int
        self.pad_char = pad_char
        self.model_size = model_size
        if model_size == 'large':
            shapes = [256, 384, 768]
        elif model_size == 'small':
            shapes = [192, 256, 512]

        self.embedding = nn.Embedding(len(char2int), char_emb_size)
        self.conv1 = nn.Conv1d(self.n_inputs, 64, kernel, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel, stride=1)
        self.conv3 = nn.Conv1d(128, shapes[0], kernel, stride=1)
        self.conv4 = nn.Conv1d(shapes[0], shapes[1], 3, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        # self.fc1 = nn.Linear(shapes[2], 384)
        self.fc1 = nn.Linear(shapes[2], output_size)

        self.activation = nn.ReLU()

        self.norm0 = nn.LayerNorm([self.n_inputs, char_emb_size])
        self.norm1 = nn.LayerNorm([64, 30])
        self.norm2 = nn.LayerNorm([128, 13])
        self.norm3 = nn.LayerNorm([shapes[0], 4])
        self.norm4 = nn.LayerNorm(shapes[2])
        # self.norm5 = nn.LayerNorm(384)

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

        # x = self.activation(x)
        # x = self.dropout(x)
        # x = self.norm4(x)
        # x = self.fc1(x)

        return x
        
    def forward(self, x):
        x = self.embed(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm4(x)

        x = self.fc1(x)
        return x

    def init_model(self, model_name=None):

        if model_name != None:
            checkpoint = torch.load(model_name)
            self.load_state_dict(checkpoint['model_state_dict'])

def create_am_distill_emb(charset_path, dropout=0.0, model_size='large', study=False):
    char2int, _ = lib.build_charset(charset_path, 0)
    if study:
        class_name = DistillEmb_T
    else:
        class_name = DistillEmb
    model = class_name(char2int,  output_size=300, pad_char=' ', dropout=dropout, model_size=model_size)
    return model

# m = create_am_distill_emb('data/am-charset.txt', model_size='large')
# x = torch.tensor(np.ones((10, 13), dtype=np.long))
# m(x)

class DistillEmb_T(nn.Module):
    
    def __init__(self, char2int, output_size, pad_char, char_emb_size=64, kernel=5, dropout=0.1, model_size='small'):
        super(DistillEmb_T, self).__init__()

        self.n_inputs = 13 # number of characters
        self.output_size = output_size
        self.char2int = char2int
        self.pad_char = pad_char
        self.model_size = model_size
        if model_size == 'large':
            shapes = [256, 384, 768]
        elif model_size == 'small':
            shapes = [192, 256, 512]

        self.embedding = nn.Embedding(len(char2int), char_emb_size)
        self.conv1 = nn.Conv1d(self.n_inputs, 64, kernel, stride=1)
        self.conv2 = nn.Conv1d(64, 128, kernel, stride=1)
        self.conv3 = nn.Conv1d(128, shapes[0], kernel, stride=1)
        self.conv4 = nn.Conv1d(shapes[0], shapes[1], 3, stride=1)
        self.pool = nn.MaxPool1d(2, 2)
        # self.fc1 = nn.Linear(shapes[2], 384)
        self.fc1 = nn.Linear(shapes[2], output_size)

        self.activation = nn.ReLU()

        self.norm0 = nn.LayerNorm([self.n_inputs, char_emb_size])
        self.norm1 = nn.LayerNorm([64, 30])
        self.norm2 = nn.LayerNorm([128, 13])
        self.norm3 = nn.LayerNorm([shapes[0], 4])
        self.norm4 = nn.LayerNorm(shapes[2])
        # self.norm5 = nn.LayerNorm(shapes[2])

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
        xc = x.view(x.shape[0], -1)
        # x = xc
        # x = self.activation(x)
        # x = self.dropout(x)
        # x = self.norm4(x)
        # x = self.fc1(x)

        return x, xc
        
    def forward(self, x):
        x1, xc = self.embed(x)
        x = x1
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm5(x)

        x = self.fc2(x)
        return x, x1, xc

    def init_model(self, model_name=None):

        if model_name != None:
            checkpoint = torch.load(model_name)
            self.load_state_dict(checkpoint['model_state_dict'])
