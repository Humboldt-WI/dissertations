import numpy as np
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import datetime
import sklearn.model_selection as sk
import torch.nn as nn

import math, copy, time
from torch.autograd import Variable


#look_back = 8
#np.random.seed(7)


"""    TRANSFORMER MODULES    """

''' Classes from this section are in poart adopted from external Repo-2021, 
    modified and extended to this experiments requirements.
    The original version can be found in Repo-2021 by Rubens Zimbres:
    https://github.com/RubensZimbres/Repo-2021/tree/main/Transformer'''

class Transformer(nn.Module):
    """
    Transformer module of a standard two tier encoder-decoder structure.
    For information on the transformer architecture refer to chapter 'Basic Transformer Archiecture'
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, out_generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.out_generator = out_generator

    def forward(self, src, tgt, src_mask, tgt_mask, src_day, tgt_day):
        '''

        :param src: [array] encoder input
        :param tgt: [array] decoder activation
        :param src_mask, tgt_mask:  [array] attention masks to mask future elements in attention
        :param src_day, tgt_day: [int] first day in inputs (to match timeline to sequences)
        :return:    [array] transformer output (prediction)
        '''
        return self.decode(self.encode(src, src_mask, src_day), src_mask,
                           tgt, tgt_mask, tgt_day)

    def encode(self, src, src_mask, src_day):
        ''' Encoder stack of transformer. '''
        return self.encoder(self.src_embed({"x": src, "x_day": src_day}), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, tgt_day):
        ''' Decoder stack of transformer. '''
        return self.decoder(self.tgt_embed({"x": tgt, "x_day": tgt_day}), memory, src_mask, tgt_mask)

import torch.nn.functional as F

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab, d_wind):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, d_wind)

    def forward(self, x):
        return F.relu(self.proj(x))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    ''' Construct a layernorm module.
        Layer normalization for various transformer sublayers using standardization.
    '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('float32')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    ''' Compute standard scaled dot product attention.
        For details see chapter 'Attention as Transformer Core Mechanism'.

    :param query:   [array] query sequences
    :param key:     [array] ke sequences
    :param value:   [array] value sequences
    :param mask:    [array] future mask
    :param dropout: [int] dropout prob.
    :return:        [arra] attention
    '''

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def sparseAttention(query, key, value, mask=None, dropout=None, att_cons=[5,3,1,0]):

    ''' Compute sparse scaled dot product attention.
        For details see chapter 'Attention as Transformer Core Mechanism'.

    :param query:   [array] query sequences
    :param key:     [array] ke sequences
    :param value:   [array] value sequences
    :param mask:    [array] future mask
    :param dropout: [float] dropout prob.
    :param att_cons:[list] sparse list of sequence elements to be included in attention connections.
                    For visualization refer to Annex E.
    :return:        [array] attention
    '''
    if att_cons == 0: att_cons = key.sisze(-2)
    n_att = len(att_cons)
    d_k = query.size(-1)
    dq0,dq1,dq2,dq3,dq4 = query.shape
    dk0,dk1,dk2,dk3,dk4 = key.shape
    sparse_query = query.reshape((dq0,dq1,dq2,dq3,1,dq4)).to(device)
    sparse_key = torch.ones((dk0,dk1,dk2,dq3,n_att,dk4)).to(device)
    for q_row in range(dq3):
        connections = np.array(att_cons)
        connections[connections > q_row] = q_row
        key_element = key[:, :, :, (q_row - connections), :]
        sparse_key[:,:,:,q_row,:,:] = key_element
    sparse_scores = torch.matmul(sparse_query, sparse_key.transpose(-2,-1)) / math.sqrt(n_att)

    scores = torch.Tensor.new_full(torch.ones(1),(dq0,dq1,dq2,dq3,dk3), -1e9).to(device)
    for q_row in range(dq3):
        connections = np.array(att_cons)
        connections[connections > q_row] = 0
        scores[:, :, :, q_row, (q_row - connections)] = sparse_scores[:,:,:,q_row,:,:].squeeze(dim=-2)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, attention_type, dropout=0.1, convolve=0, d_time=1):
        ''' MHA block. Includes convolution layer for convolutional attention and switch for canonical or sparse attention.
            For details see chapter 'Adjustments to Basic Transformer Model'

        :param h:               [int] number of parallel attention heads
        :param d_model:         [int] number of model dimensions
        :param attention_type:  [none or list] canonical or sparse attention + attention connections
        :param dropout:         [float] dropout prob.
        :param convolve:        [int] kernel size for convolution layer (def: 0 -> no convolution layer)
        :param d_time:          [int] number of sequences/channels for convolution layer
        '''

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.attention_type = attention_type
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        if convolve != 0:
            self.conv = nn.Conv1d(in_channels=d_time, out_channels=d_time, kernel_size=convolve,
                                padding=[convolve - 1], padding_mode='replicate', stride=1)#.to(self._device)
        else: self.conv = None


    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(-3)
        nbatches = query.size(1) #this nbatches is called d_time by me
        d_stock = query.size(0)
        d_wind = query.size(2)

        query, key, value = \
            [l(x).view(d_stock, nbatches, -1, self.h, self.d_k).transpose(2, 3)
             for l, x in zip(self.linears, (query, key, value))] #added d_stock, tronspose from (1,2) to (2,3)

        if self.conv != None:
            a,b,c,d,e = query.shape
            if b > 1:
                query = torch.cat(query.squeeze(dim=-1).chunk(c,dim=2),dim=0).squeeze()
                query = self.conv(query)[:,:,:d]
                query = torch.stack(query.chunk(c, dim=0),dim=2).reshape((a,b,c,d,e))
            elif b == 1:
                query = torch.cat(query.reshape((a,b,c,d)).chunk(c, dim=2), dim=0).reshape((a*c,1,d))
                query_2 = query
                for i in range(self.conv.in_channels - 1):
                    query_2 = torch.cat((query_2,query),dim=1)
                query = self.conv(query_2)[:, 0, :d].reshape((a*c,1,d))
                query = torch.stack(query.chunk(c, dim=0), dim=2).reshape((a, b, c, d, e))
            a, b, c, d, e = key.shape
            if b > 1:
                key = torch.cat(key.squeeze(dim=-1).chunk(c, dim=2), dim=0).squeeze()
                key = self.conv(key)[:, :, :d]
                key = torch.stack(key.chunk(c, dim=0), dim=2).reshape((a, b, c, d, e))
            elif b == 1:
                key = torch.cat(key.reshape((a,b,c,d)).chunk(c, dim=2), dim=0).reshape((a*c,1,d))
                key_2 = key
                for i in range(self.conv.in_channels - 1):
                    key_2 = torch.cat((key_2,key),dim=1)
                key = self.conv(key_2)[:, 0, :d].reshape((a*c,1,d))
                key = torch.stack(key.chunk(c, dim=0), dim=2).reshape((a,b,c,d,e))

        if self.attention_type == 'full':
            x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        else:
            x, self.attn = sparseAttention(query, key, value, mask=mask, dropout=self.dropout, att_cons=self.attention_type )

        x = x.transpose(2, 3).contiguous() \
            .view(d_stock, nbatches, -1, self.h * self.d_k) #added d_stock, transpose from (1,2) to (2,3)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    ''' Feedforward network sublayer. '''

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings1(nn.Module):
    ''' Implements token embedding for encoder stack. '''
    def __init__(self, d_model, vocab):
        super(Embeddings1, self).__init__()
        self.d_model = d_model

    def forward(self, x_dict):
        x = x_dict["x"]
        x_day = x_dict["x_day"]
        d_stock, d_time, d_wind, d_var = x.shape

        return {"x": torch.stack(x.repeat(self.d_model, 1, 1, 1).chunk(chunks=self.d_model, dim=0), dim=3).reshape((d_stock, d_time, d_wind, self.d_model*d_var)) * math.sqrt(self.d_model), "x_day": x_day} #unbeding wieder hintert dim=3) händen#* math.sqrt(self.d_model)
        #last to dims where a,b four times below. after reshape a,b,a,b,a,b,a,b. when later reshaped again it reverses to a,b four times again


class Embeddings2(nn.Module):
    ''' Implements token embedding for decoder stack. '''
    def __init__(self, d_model, vocab):
        super(Embeddings2, self).__init__()
        self.d_model = d_model

    def forward(self, x_dict):
        x = x_dict["x"]
        x_day = x_dict["x_day"]
        d_stock, d_time, d_wind, d_var = x.shape

        return {"x": torch.stack(x.repeat(self.d_model, 1, 1, 1).chunk(chunks=self.d_model, dim=0), dim=3).reshape((d_stock, d_time, d_wind, self.d_model*d_var)) * math.sqrt(self.d_model), "x_day": x_day}  # * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    ''' Implements positional encoding.
        For details refer to chapter "Richer Positional Encoding"'''

    def __init__(self, d_model, dropout, pe_type, timeline, max_len=8218):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_type = pe_type

        if pe_type == None:
            pe = torch.zeros(max_len, d_model)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        if pe_type == 'relative' or pe_type == 'global':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                 -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        if pe_type == 'weekday':
            pe = torch.zeros(max_len, d_model)
            date = []
            weekday = []
            for i in range(len(timeline)):
                date.append(np.datetime64(timeline[i]).astype(datetime.datetime))
                weekday.append(date[-1].weekday())
            position = torch.Tensor(weekday).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                 -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        if pe_type == 'yearday':
            pe = torch.zeros(max_len, d_model)
            date = []
            yearday = []
            for i in range(len(timeline)):
                date.append(np.datetime64(timeline[i]).astype(datetime.datetime))
                yearday.append(date[-1].timetuple().tm_yday)
            position = torch.Tensor(yearday).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) *
                                 -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

    def forward(self, x_dict):
        x = x_dict["x"]
        x_day = x_dict["x_day"]
        a, b, c, d = x.shape
        if self.pe_type == None:
            pass
        if self.pe_type == 'relative':
            # Every time series input gets a relative position encoding 0 to k.(0,1,2,3,4,...)
            x = x + self.pe[:,:c,:]
        if self.pe_type == 'global':
            # Every time series input gets a global position encoding j to k. (relative in respect to complete dataset. 5,6,...)
            pe = self.pe[:,x_day:x_day + c + b - 1,:]
            idx = []
            for i in range(b):
                idx.append(range(i,i+c))
            pe = pe[:,idx,:]
            x = x + pe
        if self.pe_type == 'weekday':
            # Every time series input gets a periodic position encoding based on day of week of inputs. (Mon=0, Tue=1...)
            pe = self.pe[:, x_day:x_day + c + b - 1, :]
            idx = []
            for i in range(b):
                idx.append(range(i, i + c))
            pe = pe[:, idx, :]
            x = x + pe
        if self.pe_type == 'yearday':
            # Every time series input gets a periodic position encoding based on calendar number of day in year.(01.01.=0, 02.01.=1...)
            pe = self.pe[:, x_day:x_day + c + b - 1, :]
            idx = []
            for i in range(b):
                idx.append(range(i, i + c))
            pe = pe[:, idx, :]
            x = x + pe
        return x


def make_model(src_vocab, tgt_vocab, d_wind, d_var, d_tgt, timeline, N=2,
               d_model=4, d_ff=32, h=4, dropout=0.1, convolve=0, d_time=1, pe_type='relative', attention_type='full'):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h*d_var, d_model*d_var, attention_type, dropout, convolve, d_time)
    ff = PositionwiseFeedForward(d_model*d_var, d_ff, dropout)
    position = PositionalEncoding(d_model*d_var, dropout, pe_type, timeline)
    model = Transformer(
        Encoder(EncoderLayer(d_model*d_var, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model*d_var, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings1(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings2(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab, d_wind),
        Generator(d_var, d_tgt, d_tgt)).to(device)


    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0, X_day=0, Y_day=0, var_order=None):

        if trg is not None:
            '''
            self.src = src[:, :, :, :-1]
            self.src_mask = (src[:, :, :, 0] != pad).unsqueeze(-2)
            self.X_day = X_day
            '''
            if var_order[-1] in ['ret','price']:
                self.src = src[:, :, :, :-1]
                self.src_mask = (src[:, :, :, 0] != pad).unsqueeze(-2)
                self.X_day = X_day

                self.trg = trg[:, :, :-1, :-1]
                self.trg_y = trg[:, :, 1:, -1]
                self.ntokens = (self.trg_y != pad).data.sum(dim=(-2, -1))[0]
            elif var_order[-1] in ['binclass']:
                #binclass is one-hot encoded
                self.src = src[:, :, :, :-2]
                self.src_mask = (src[:, :, :, 0] != pad).unsqueeze(-2)
                self.X_day = X_day

                self.trg = trg[:, :, :-1, :-2]
                self.trg_y = trg[:, :, 1:, -2]
                self.ntokens = (self.trg_y != pad).data.sum(dim=(-2, -1))[0]
            self.trg_mask = \
                self.make_std_mask(self.trg[:,:,:,0], pad)

            self.Y_day = Y_day

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future days."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute, learning, nbatch, d_stock):
    "Standard Training and Logging Function"
    train_idx, val_idx = sk.train_test_split(np.array(range(d_stock)),test_size=0.1)
    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_val_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src.to(device), batch.trg.to(device),
                            batch.src_mask.to(device), batch.trg_mask.to(device), batch.X_day, batch.Y_day).to(device)
        a, b, c, d = batch.trg.shape
        if i in train_idx:
            loss = loss_compute(out.reshape((a, b, c, -1, d)).permute(0,1,2,4,3), batch.trg_y, batch.ntokens, learning).to(device)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i == nbatch - 1:
                elapsed = time.time() - start
                print("Epoch Steps: %d Loss: %f Tokens per Sec: %f" %
                      (i, loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        elif i in val_idx:
            val_loss = loss_compute(out.reshape((a, b, c, -1, d)).permute(0, 1, 2, 4, 3), batch.trg_y, batch.ntokens,
                                learning, update=False).to(device)
            total_val_loss += val_loss
    return total_loss, total_val_loss  # / total_tokens


global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self, learning):
        "Update parameters and rate"
        self._step += 1
        for p in self.optimizer.param_groups:
            p['lr'] = learning
        self._rate = learning
        self.optimizer.step()


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-9))


def data_gen(V, batch, nbatches, X0, Y0, X_day, Y_day, var_order):
    "Generate random data for a src-tgt copy task."
    d_stock, d_time, d_wind, d_var = X0.shape
    batch_size = int(d_stock / nbatches)
    for i in range(nbatches):
        data1 = torch.from_numpy(X0.reshape(d_stock, d_time, d_wind, d_var))
        data2 = torch.from_numpy(Y0.reshape(d_stock, d_time, d_wind, d_var))
        data1 = data1[(batch_size * i):(min((batch_size * (i + 1)),d_stock))]
        data2 = data2[(batch_size * i):(min((batch_size * (i + 1)), d_stock))]
        src = Variable(data1, requires_grad=False)
        tgt = Variable(data2, requires_grad=False)
        yield Batch(src, tgt, 0, X_day, Y_day, var_order)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, out_generator, criterion, opt=None):
        self.generator = generator
        self.out_generator = out_generator
        self.criterion = criterion
        self.opt = opt


    def __call__(self, x, y, norm, learning, update=True):
        d_stock, d_time, d_wind, d_var, _ = x.shape
        x = self.generator(x)
        x = self.out_generator(x.permute(0,1,2,4,3)).permute(0,1,2,4,3)

        d_stock, d_time, d_wind, d_var, _ = x.shape
        x = torch.sum(x.reshape(d_stock, d_time, d_wind, d_var, -1), (-1))
        x = x.reshape((d_stock, d_time, d_wind))
        loss = self.criterion(x[:,:,-1].to(device),y[:,:,-1].to(device))  # sum over days #for colab both at dim -2)

        if update == True:
            if loss < 0.01:
                learning = learning / 3

            loss.backward()
            if self.opt is not None:
                self.opt.step(learning)
                self.opt.optimizer.zero_grad()
        return loss.data  # * norm

def m_init_switch(m_init_freq, m_init_counter):
    """ Checks whether model shall be recompiled.

    :param m_init_freq:     Compilation frequency
    :param m_init_counter:  Experiment step.
    :return:
    """
    if m_init_freq == 0 and m_init_counter == 0:
        return True
    if m_init_freq == 1:
        return True
    if m_init_freq > 1 and m_init_counter % m_init_freq == 0:
        return True
    else:
        return False

def greedy_decode(model, src, src_mask, max_len, start_symbol, x_day=0, y_day=0):
    """ Legacy function that can be used for prediction instead of a full decoder stack.
        Support discontinued, included for completeness only.
    :param model:
    :param src:
    :param src_mask:
    :param max_len:
    :param start_symbol:
    :param x_day:
    :param y_day:
    :return:
    """
    memory = model.encode(src.to(device), src_mask.to(device), x_day)
    d_stock, _, d_wind, d_var = src.shape

    ys = torch.ones(d_stock, 1, d_wind-1, d_var).fill_(start_symbol).type_as(src.data)
    for i in range(1): #(d_wind - 1)
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(-2)).type_as(src.data)), y_day)
        prob = torch.sum(src * torch.sum(model.generator(out.reshape((d_stock, 1, d_wind-1,d_var,-1))), (-3)).permute(0,1,3,2),((-2,-1)))

    return prob


