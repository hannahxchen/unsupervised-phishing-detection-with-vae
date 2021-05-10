import os
import json
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from functools import partial

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def url_iterator(filename='datasets/benign_train.csv'):
    df = pd.read_csv(filename)
    for url in df.URL:
        yield url

def train_tokenizer():
    # First we create an empty Byte-Pair Encoding model (i.e. not trained model)
    tokenizer = Tokenizer(BPE())

    # Then we enable lower-casing and unicode-normalization
    # The Sequence normalizer allows us to combine multiple Normalizer that will be
    # executed in order.
    tokenizer.normalizer = Sequence([
        NFKC(),
        Lowercase()
    ])

    # Our tokenizer also needs a pre-tokenizer responsible for converting the input to a ByteLevel representation.
    tokenizer.pre_tokenizer = ByteLevel()

    # And finally, let's plug a decoder so we can recover from a tokenized input to the original one
    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.enable_padding()

    # We initialize our trainer, giving him the details about the vocabulary we want to generate
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"], show_progress=True, initial_alphabet=ByteLevel.alphabet())
    tokenizer.train_from_iterator(url_iterator(), trainer=trainer)

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))

    tokenizer.model.save('./tokenizer')

    return tokenizer

class UrlDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename).URL.tolist()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def get_dataset(data_dir='datasets', data_split='train'):
    if data_split == 'train':
        return UrlDataset(os.path.join(data_dir, 'benign_train.csv'))
    elif data_split == 'dev':
        return UrlDataset(os.path.join(data_dir, 'benign_dev.csv'))
    elif data_split == 'test':
        return UrlDataset(os.path.join(data_dir, 'test.csv'))
    else:
        raise ExceptionError('Data split error')
        
def collate(examples, tokenizer):
    inputs = ['[SOS]' + e for e in examples]
    inputs = torch.tensor([ex.ids for ex in tokenizer.encode_batch(inputs)])
    targets = [e + '[EOS]' for e in examples]
    targets = torch.tensor([ex.ids for ex in tokenizer.encode_batch(targets)])
    return inputs, targets

def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))

class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.nonlinear):
            self._add_to_parameters(module.parameters(), 'nonlinear_module_{}'.format(i))

        self.linear = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.linear):
            self._add_to_parameters(module.parameters(), 'linear_module_{}'.format(i))

        self.gate = [nn.Linear(size, size) for _ in range(num_layers)]
        for i, module in enumerate(self.gate):
            self._add_to_parameters(module.parameters(), 'gate_module_{}'.format(i))

        self.f = f

    def forward(self, x):
        """
        :param x: tensor with shape of [batch_size, size]
        :return: tensor with shape of [batch_size, size]
        applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
        """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x

    def _add_to_parameters(self, parameters, name):
        for i, parameter in enumerate(parameters):
            self.register_parameter(name='{}-{}'.format(name, i), param=parameter)

class VAE(nn.Module):
    def __init__(self, vocab_size, embed_size, encoder_hidden_size, decoder_hidden_size,
        dropout_rate, latent_size, encoder_num_layers, decoder_num_layers, word_dropout, sos_idx, unk_idx, pad_idx):
        super().__init__()
        
        self.latent_size = latent_size
        self.word_dropout_rate = word_dropout
        self.encoder_num_layers = encoder_num_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.sos_idx = sos_idx
        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        # self.kernels = [(1, 25), (2, 50), (3, 75), (4, 100), (5, 125), (6, 150)]
        # self.sum_depth = fold(lambda x, y: x + y, [depth for _, depth in self.kernels], 0)
        
        # self.highway = Highway(embed_size, 2, F.relu)
        self.encoder = nn.LSTM(embed_size, encoder_hidden_size, num_layers=encoder_num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(latent_size+embed_size, decoder_hidden_size, num_layers=decoder_num_layers, batch_first=True)
        
        self.mu = nn.Linear(encoder_hidden_size*2, latent_size)
        self.logvar = nn.Linear(encoder_hidden_size*2, latent_size)
        self.fc = nn.Linear(decoder_hidden_size, vocab_size)
        
    def encode(self, encoder_input, batch_size, seq_lens):
        packed = pack_padded_sequence(encoder_input, seq_lens, batch_first=True, enforce_sorted=False)
        encoded_output, (_, final_state) = self.encoder(packed)

        final_state = final_state.view(self.encoder_num_layers, 2, batch_size, self.encoder_hidden_size)
        final_state = final_state[-1]
        h1, h2 = final_state[0], final_state[1]
        final_state = torch.cat([h1, h2], 1)
        
        return final_state
    
    def reparameterize(self, batch_size, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            z = Variable(torch.randn([batch_size, self.latent_size])).to(device)
            return z*std + mu
        else:
            return mu
        
    def decode(self, decoder_input, z, batch_size, seq_len):
        decoder_input = self.embedding(decoder_input)
        decoder_input = self.embedding_dropout(decoder_input)
        
        z = torch.cat([z]*seq_len, 1).view(batch_size, seq_len, self.latent_size)
        decoder_input = torch.cat([decoder_input, z], 2)
        decoded_output, hidden = self.decoder(decoder_input)
        
        return self.fc(decoded_output)
    
    def forward(self, inputs):
        
        embeds = self.embedding(inputs)
        embeds = self.embedding_dropout(embeds)
        [batch_size, seq_len, embed_size] = embeds.size()
        
        # encoder_input = embeds.view(-1, embed_size)
        # encoder_input = self.highway(encoder_input)
        # encoder_input = encoder_input.view(batch_size, seq_len, embed_size)
        
        orig_seq_len = [len([w for w in example if w != 0]) for example in inputs]
        hidden = self.encode(embeds, batch_size, orig_seq_len).squeeze()
        
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)
        z = self.reparameterize(batch_size, mu, logvar)
        
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(inputs.size()).to(device)
            prob[(inputs.data - self.sos_idx) * (inputs.data - self.pad_idx) == 0] = 1
            decoder_input = inputs.clone()
            decoder_input[prob < self.word_dropout_rate] = self.unk_idx

        decoder_output = self.decode(decoder_input, z, batch_size, seq_len)
        
        return decoder_output, mu, logvar

def kld_coef(i):
    import math
    return (math.tanh((i - 3500)/1000) + 1)/2

def loss_func(recon_x, x, mu, logvar):
    recon_loss = F.cross_entropy(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.div(KLD, x.shape[0])
    return recon_loss, KLD

def train(model, train_dataloader, optimizer, iteration, vocab_size):
    model.train()
    train_loss = 0
    recon_loss_all = 0
    KLD_loss_all = 0
    
    for batch in tqdm(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        
        inputs, targets = batch
        recon_batch, mu, logvar = model(inputs)
        
        recon_loss, KLD = loss_func(recon_batch.view(-1, vocab_size), targets.reshape(targets.shape[0]*targets.shape[1]), mu, logvar)
        loss = recon_loss + kld_coef(iteration) * KLD
        
        optimizer.zero_grad()
        loss.backward()
        
        train_loss += loss.item()
        recon_loss_all += recon_loss.item()
        KLD_loss_all += KLD.item()
        
        optimizer.step()
        iteration += 1
        
    return train_loss, recon_loss_all, KLD_loss_all, iteration

def evaluate(model, dev_dataloader, vocab_size):
    model.eval()
    recon_loss_all = 0
    kld_loss_all = 0
    
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            batch = tuple(t.to(device) for t in batch)
        
            inputs, targets = batch
            recon_batch, mu, logvar = model(inputs)

            recon_loss, KLD = loss_func(recon_batch.view(-1, vocab_size), targets.reshape(targets.shape[0]*targets.shape[1]), mu, logvar)
            recon_loss_all += recon_loss.item()
            kld_loss_all += KLD.item()
            
    return recon_loss_all, kld_loss_all

def main():
    tokenizer = train_tokenizer()

    train_dataset = get_dataset()
    dev_dataset = get_dataset(data_split='dev')
    train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=partial(collate, tokenizer=tokenizer), shuffle=True, pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=partial(collate, tokenizer=tokenizer), shuffle=False, pin_memory=True)

    learning_rate = 1e-3
    epochs = 30
    vocab_size = tokenizer.get_vocab_size()

    model = VAE(vocab_size=tokenizer.get_vocab_size(), embed_size=300, encoder_hidden_size=500, decoder_hidden_size=500,
                dropout_rate=0.5, latent_size=100, encoder_num_layers=1, decoder_num_layers=1, word_dropout=0.5, 
                sos_idx=tokenizer.token_to_id("[SOS]"), unk_idx=tokenizer.token_to_id("[UNK]"), pad_idx=tokenizer.token_to_id("[PAD]"))

    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # train_loss_all = []
    # recon_loss_all = []
    # kld_loss_all = []
    # val_recon_loss_all = []
    # val_kld_loss_all = []
    iteration = 0

    for e in tqdm(range(epochs)):
        train_loss, recon_loss, kld_loss, iteration = train(model, train_dataloader, optimizer, iteration, vocab_size)

        # train_loss_all.append(train_loss)
        # recon_loss_all.append(recon_loss)
        # kld_loss_all.append(kld_loss)
        
        val_recon_loss, val_kld_loss = evaluate(model, dev_dataloader, vocab_size)
        # val_recon_loss_all.append(val_recon_loss)
        # val_kld_loss_all.append(val_kld_loss)

        results = {'epoch': e, 
                   'iteration': iteration,
                   'train_loss': train_loss, 
                   'train_recon_loss': recon_loss, 
                   'train_kld_loss': kld_loss, 
                   'dev_recon_loss': val_recon_loss, 
                   'dev_kld_loss': val_kld_loss}

        if (e+1)%10 == 0:
            torch.save(model.state_dict(), f'vae_model_3_epoch_{e}')

        with open('eval_metrics.json', "a") as writer:
            writer.write(json.dumps(results) + "\n")

        print(f'Epoch[{e}/{epochs}], train_recon_loss: {recon_loss}, train_kld_loss: {kld_loss}, dev_recon_loss: {val_recon_loss}, dev_kld_loss: {val_kld_loss}')

    torch.save(model.state_dict(), 'vae_model_4')


if __name__ == "__main__":
    main()