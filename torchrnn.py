import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# One hot encode
def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

# Batches
def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size n_seqs x n_steps from arr.
    
        Args
        ----
        arr: Array to make batches from
        n_seqs: Batch size, number of sequences per batch
        n_steps: Number of sequence steps per batch
    '''
    batch_size = n_seqs * n_steps
    n_batches = len(arr)//batch_size
    
    arr = arr[:n_batches * batch_size]
    arr = arr.reshape((n_seqs, -1))
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y = np.zeros_like(x)
        
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
        

class CharRNN(nn.Module):
    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2, 
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))
        self.init_weights()
        
        
    def forward(self, x, hc):
        ''' Forward pass. Inputs are `x` and hidden/cell state are `hc`. '''
        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)
        x = x.reshape(x.size()[0]*x.size()[1], self.n_hidden)
        x = self.fc(x)
        return x, (h, c)
    
    
    def predict(self, device, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns predicted character and hidden state.
        '''
        self.to(device) 
        if h is None:
            h = self.init_hidden(1)
            
        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        inputs = torch.from_numpy(x).to(device)
        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)
        p = F.softmax(out, dim=1).data.cpu()
        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        return self.int2char[char], h
    
    def init_weights(self):
        ''' Initialize weights for fc layer '''
        initrange = 0.1
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)
        
    def init_hidden(self, n_seqs):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())
        

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False              
    

# Train the model
training_df = pd.DataFrame(columns=['corpus', 'epoch', 'step', 'loss', 'val_loss', 'val_perplexity'])

def train(lang, net, device, data, epochs=10, n_seqs=10, n_steps=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network
    
        Args
        ----
        lang: Current corpus
        net: CharRNN network
        device: CPU or GPU (cuda)
        data: text data to train on
        epochs: Number of epochs to train
        n_seqs: Number of sequences per batch (batch size)
        n_steps: Number of character steps per batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Validation data as fraction
        print_every: Number of steps for printing training/validation loss
    '''
    global training_df
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    net.to(device)
    counter = 0
    n_chars = len(net.chars)
    results = []
    for e in range(epochs):
        h = net.init_hidden(n_seqs)
        for x, y in get_batches(data, n_seqs, n_steps):
            counter += 1
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            h = tuple([each.data for each in h])
            net.zero_grad()
            output, h = net.forward(inputs, h)
            loss = criterion(output, targets.view(n_seqs*n_steps).type(torch.LongTensor).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            if counter % print_every == 0:
                val_h = net.init_hidden(n_seqs)
                val_losses = []
                perplexities = []
                for x, y in get_batches(val_data, n_seqs, n_steps):
                    x = one_hot_encode(x, n_chars)
                    inputs, targets = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                    val_h = tuple([each.data for each in val_h])
                    output, val_h = net.forward(inputs, val_h)
                    val_loss = criterion(output, targets.view(n_seqs*n_steps).type(torch.LongTensor).to(device))
                    val_perplexity = torch.exp(val_loss)
                    val_losses.append(val_loss.item())
                    perplexities.append(val_perplexity.item())
                    
                print(
                    "Epoch: {}/{}...".format(e+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.4f}...".format(loss.item()),
                    "Val Loss: {:.4f}...".format(np.mean(val_losses)),
                    "Val Perplexity: {:.4f}".format(np.mean(perplexities))
                )
    
                results.append({
                    'corpus': lang,
                    'epoch': e + 1,
                    'step': counter,
                    'loss': loss.item(),
                    'val_loss': np.mean(val_losses),
                    'val_perplexity': np.mean(perplexities)
                })
                
    temp = pd.DataFrame(results)
    training_df = pd.concat([training_df, temp], ignore_index=True)

                
# Hyperparams
n_seqs = 128
n_steps = 100
epochs = 20
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maindir = './current_corpora/'
count = 1

if __name__ == '__main__':
    for file in os.listdir(maindir):
        if file.endswith('.txt'):
            lang = os.path.splitext(os.path.basename(file))[0].split('_')[0]
            with open(os.path.join(maindir, file), 'r', encoding='utf-8') as f:
                text = f.read()
            f.close() 

            print(f'Starting {file}...')
            print(f'{count}/22')
            count += 1
            chars = tuple(set(text))
            int2char = dict(enumerate(chars))
            char2int = {ch: ii for ii, ch in int2char.items()}
            encoded = np.array([char2int[ch] for ch in text])
            net = CharRNN(chars, n_hidden=512, n_layers=2)
            train(lang, net, device, encoded, epochs=epochs, n_seqs=n_seqs, n_steps=n_steps, lr=lr, clip=5)
            
    with open('./torch_rnn_results.csv', mode='w') as f:
        training_df.to_csv(f)