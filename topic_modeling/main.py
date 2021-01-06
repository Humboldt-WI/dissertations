"""Parts of this file are obtained from https://github.com/adjidieng/DETM/blob/master/main.py, 10/2020,
several changes were made to add functionality and remove bugs."""

import argparse
from collections import Counter
from datetime import datetime 
import numpy as np 
import math 
import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import pandas as pd 
import pickle 
import pytz
import random 
import seaborn as sns
import scipy.io
from sklearn.decomposition import PCA
import sys
import time 
import torch
from torch import nn, optim
from torch.nn import functional as F

from detm import DETM
from utils import get_data, get_batch, get_rnn_input # data
from utils import get_eta, eta_helper, get_theta, get_final_theta # results
from utils import get_topic_coherence, get_completion_ppl, diversity_helper, get_topic_quality # evaluation
from utils import nearest_neighbors, cosine_neighbors # neighbours
from utils import get_topic_labels, get_topic_labels_by_frequency, get_topic_labels_by_similarity, get_topic_labels_from_beta  # topic labels
from utils import visualize #prints, visualisation of topics, word embeddings

parser = argparse.ArgumentParser(description='The Dynamic Embedded Topic Model') 

### model related arguments
parser.add_argument('--num_topics', type=int, default=50, help='number of topics')
parser.add_argument('--delta', type=float, default=0.005, help='prior variance')
parser.add_argument('--rho_size', type=int, default=200, help='dimension of word embeddings')
#parser.add_argument('--emb_size', type=int, default=200, help='dimension of embeddings')
parser.add_argument('--train_embeddings', type=int, default=0, help='train embeddings (1) or use pre-trained embs (0)')
parser.add_argument('--theta_hidden_size', type=int, default=800, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu', help='activation function for theta NN')
parser.add_argument('--eta_nlayers', type=int, default=4, help='number of layers for eta RNN')
parser.add_argument('--eta_hidden_size', type=int, default=400, help='number of hidden units for eta RNN')

### training and evaluation related arguments
parser.add_argument('--version', type=str, default='', help='name of model version')
parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--epochs', type=int, default=1000, help='(maximum) number of epochs')
parser.add_argument('--early_stopping', type=int, default=1, help='stop early (1) or not (0)')
parser.add_argument('--patience', type=int, default=25, help='patience for early stopping (number of epochs)')
parser.add_argument('--batch_size', type=int, default=256, help='number of documents in a batch')
parser.add_argument('--increase_batch_size', type=int, default=0, help='increase the batch size during training (1) or not (0)')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimiser (adam, adamw,sgd)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--anneal_lr', type=int, default=0, help='anneal the learning rate (1) or not')
parser.add_argument('--bad_hits', type=int, default=10, help='number of bad hits allowed (anneal lr)')
parser.add_argument('--lr_factor', type=float, default=0.9, help='multiply learning rate by this when annealing')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')
parser.add_argument('--nll_weight', type=float, default=1.0, help='weight of rec loss')
parser.add_argument('--theta_dropout', type=float, default=0.0, help='dropout rate (theta NN)')
parser.add_argument('--eta_dropout', type=float, default=0.0, help='dropout rate (eta RNN)')
parser.add_argument('--bow_norm', type=int, default=1, help='normalise the bag-of-words (1) or not (0)')
parser.add_argument('--load_from', type=str, default='', help='model checkpoint for evaluation')
parser.add_argument('--seed', type=int, default=1, help='random seed')

### data and file related arguments
parser.add_argument('--data_path', type=str, default='Data/Technology-Data/processed/final/grouped_years/min_df_50/', help='directory containing data')
parser.add_argument('--emb_type', type=str, default='Word2Vec', help='type of word embedding (for printing and evaluation purposes only)')
parser.add_argument('--emb_path', type=str, default='Data/Embeddings/Word2Vec/Word2Vec_200.txt', help='directory containing embeddings')
parser.add_argument('--save_path', type=str, default='Results', help='path to save results')

### printing and visualisation related arguments
parser.add_argument('--log_interval', type=int, default=25, help='when to log training')
parser.add_argument('--num_words', type=int, default=10, help='number of words for topic viz')
parser.add_argument('--visualize_every', type=int, default=100, help='when to visualize results')

args = parser.parse_args()


## Settings
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pca = PCA(n_components=2)

## Get data
print('Fetch data...')
vocab, full, train, valid, test = get_data(args.data_path)

print('Get vocabulary and embeddings...')
vocab_size = len(vocab)
args.vocab_size = vocab_size
emb_path = args.emb_path
vect_path = os.path.join(args.data_path.split('/')[0], 'embeddings.pkl')   
vectors = {}
with open(emb_path, 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        if word in vocab:
            try:
                vect = np.array(line[1:]).astype(np.float)
            except ValueError:
                if args.emb_type.lower() == 'glove' and len(line[1:])>args.rho_size:
                    vect = np.array(line[2:]).astype(np.float)
            vectors[word] = vect
embeddings = np.zeros((vocab_size, args.rho_size))
words_found = 0
words_not_found = [] 
for i, word in enumerate(vocab):
    try: 
        embeddings[i] = vectors[word]
        words_found += 1
    except KeyError:
        embeddings[i] = np.random.normal(scale=0.6, size=(args.rho_size, ))
        words_not_found.append(word)

embeddings = torch.from_numpy(embeddings).to(device)
args.embeddings_dim = embeddings.size()

if len(words_not_found)>0:
    print('Number of words for which embedding was not available: ', len(words_not_found))
if len(words_not_found) == len(vocab) and args.train_embeddings != 1:
    sys.exit('Interrupt execution. Pre-trained embeddings could not be read.')

print('Get time stamps...')
with open(args.data_path + 'timestamps.pkl', 'rb') as f:
    timestamps = pickle.load(f)
    
print('Get full dataset...')
full_tokens = full['tokens']
print(full_tokens.shape)
full_counts = full['counts']
full_times = full['times']
args.num_times = len(np.unique(full_times))
args.num_docs_full = len(full_tokens)
full_rnn_inp = get_rnn_input(
    full_tokens, full_counts, full_times, args.num_times, args.vocab_size, args.num_docs_full)

print('Get training data...')
train_tokens = train['tokens']
print(train_tokens.shape)
train_counts = train['counts']
train_times = train['times']
args.num_times = len(np.unique(train_times))
args.num_docs_train = len(train_tokens)
train_rnn_inp = get_rnn_input(
    train_tokens, train_counts, train_times, args.num_times, args.vocab_size, args.num_docs_train)

print('Get validation data...')
valid_tokens = valid['tokens']
print(valid_tokens.shape)
valid_counts = valid['counts']
valid_times = valid['times']
args.num_docs_valid = len(valid_tokens)
valid_rnn_inp = get_rnn_input(
    valid_tokens, valid_counts, valid_times, args.num_times, args.vocab_size, args.num_docs_valid)

print('Get testing data...')
test_tokens = test['tokens']
print(test_tokens.shape)
test_counts = test['counts']
test_times = test['times']
args.num_docs_test = len(test_tokens)
test_rnn_inp = get_rnn_input(
    test_tokens, test_counts, test_times, args.num_times, args.vocab_size, args.num_docs_test)

test_1_tokens = test['tokens_1']
test_1_counts = test['counts_1']
test_1_times = test_times
args.num_docs_test_1 = len(test_1_tokens)
test_1_rnn_inp = get_rnn_input(
    test_1_tokens, test_1_counts, test_1_times, args.num_times, args.vocab_size, args.num_docs_test)

test_2_tokens = test['tokens_2']
test_2_counts = test['counts_2']
test_2_times = test_times
args.num_docs_test_2 = len(test_2_tokens)
test_2_rnn_inp = get_rnn_input(
    test_2_tokens, test_2_counts, test_2_times, args.num_times, args.vocab_size, args.num_docs_test)

rnn_inputs={'train':train_rnn_inp,
            'test1':test_1_rnn_inp,
            'val':valid_rnn_inp,
            'full':full_rnn_inp}

## Define model checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    
if not os.path.exists(os.path.join(args.save_path,args.version)):
    os.makedirs(os.path.join(args.save_path,args.version))
    
if args.mode == 'eval':
    ckpt = args.load_from
else:
    execution_time = datetime.now(pytz.timezone('Europe/Berlin')).strftime("%d-%m-%Y_%Hh%Mm")
    print('\nExecution (Start) Time: ', execution_time)
    ckpt = os.path.join(args.save_path, args.version,'DETM_{}_Exec_{}'.format(args.version, execution_time))
    print('\nCreated Checkpoint: {}'.format(ckpt))
args.ckpt = ckpt

## Model and optimizer
if args.load_from != '' and args.mode != 'train':
    print('Load checkpoint from: {}'.format(args.load_from))
    with open(args.load_from, 'rb') as f:
        model = torch.load(f, map_location = device)
else:
    model = DETM(args, embeddings)
    print('\nDETM architecture: {}'.format(model))
model.to(device)

if args.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

if args.mode == 'train':
    print('Optimizer:\n',optimizer)

## Training
def train(epoch):
    """Train DETM on data for one epoch.
    """
    model.train()
    acc_loss = 0
    acc_nll = 0
    acc_kl_theta_loss = 0
    acc_kl_eta_loss = 0
    acc_kl_alpha_loss = 0
    cnt = 0

    if args.increase_batch_size:
        if epoch == 0 or epoch/5 < 5:
            batch_size = 32
        elif epoch/5 < 10: 
            batch_size = 64
        elif epoch/5 < 15: 
            batch_size = 128
        elif epoch/5 < 20:
            batch_size = 256
        elif epoch/5 < 25:
            batch_size = 512
        else: 
            batch_size = 1024
            
        if batch_size <= args.batch_size:
            batch_size = batch_size
        else:
            batch_size = args.batch_size
    else:
        batch_size = args.batch_size
        
    indices = torch.randperm(args.num_docs_train)
    indices = torch.split(indices, batch_size)

    for idx, ind in enumerate(indices):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch, times_batch = get_batch(
            train_tokens, train_counts, ind, args.vocab_size, args.rho_size, times=train_times)
        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch
        
        loss, nll, kl_alpha, kl_eta, kl_theta = model(data_batch, normalized_data_batch, times_batch, train_rnn_inp, args.num_docs_train)
        
        weighted_loss = args.nll_weight*nll + kl_alpha + kl_eta + kl_theta
        weighted_loss.backward() 
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(loss).item()
        acc_nll += torch.sum(nll).item()
        acc_kl_theta_loss += torch.sum(kl_theta).item()
        acc_kl_eta_loss += torch.sum(kl_eta).item()
        acc_kl_alpha_loss += torch.sum(kl_alpha).item()
        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2) 
            cur_nll = round(acc_nll / cnt, 2) 
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
            cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
            cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
            lr = optimizer.param_groups[0]['lr']
            print('... batch {}/{} ... LR: {} ... NELBO: {}'.format(idx, len(indices), lr, cur_loss))
    
    cur_loss = round(acc_loss / cnt, 2) 
    cur_nll = round(acc_nll / cnt, 2) 
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2) 
    cur_kl_eta = round(acc_kl_eta_loss / cnt, 2) 
    cur_kl_alpha = round(acc_kl_alpha_loss / cnt, 2) 
    lr = optimizer.param_groups[0]['lr']
    print('Learning Rate: {}'.format(lr))
    print('Batch Size: {}'.format(batch_size))
    print('NELBO: {}'.format(cur_loss))
    loss_trace.append(cur_loss)
    nll_trace.append(cur_nll)
    kl_theta_trace.append(cur_kl_theta)
    kl_eta_trace.append(cur_kl_eta)
    kl_alpha_trace.append(cur_kl_alpha)
    lr_trace.append(lr)         

if args.mode == 'train':
    print('\n')
    print('=*'*100)
    print('Train a Dynamic Embedded Topic Model on Guardian Technology Data with the following settings: {}'.format(args)) 
    print('Version:', args.version)
    print('Checkpoint:', ckpt)
    print('=*'*100)
    
    lr_trace = []
    kl_theta_trace = []
    kl_eta_trace = []
    kl_alpha_trace = []
    nll_trace = []
    loss_trace = []
    
    # intialise labels
    topic_labels = None
    
    ## train model on data by looping through multiple epochs
    best_epoch = 0
    best_val_ppl = 100e10
    all_val_ppls = []

    for epoch in range(1, args.epochs+1): 
        print('*'*55)
        print('Epoch {}/{}'.format(epoch, args.epochs))
        train(epoch)
        val_ppl =  get_completion_ppl('val', valid, rnn_inputs, model, args)
        all_val_ppls.append(val_ppl)
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr:
                if len(all_val_ppls) > args.bad_hits and val_ppl > min(all_val_ppls[:-args.bad_hits]) and (args.lr_factor*lr) >= 1e-5:
                    optimizer.param_groups[0]['lr'] *= args.lr_factor
                        
        if epoch % args.visualize_every == 0 and epoch != args.epochs: 
            visualize(args, model, vocab, topic_labels, timestamps)
            
        if args.early_stopping:
            if epoch >= best_epoch + args.patience:
                print('\nEarly stopping at Epoch {}. Epoch with lowest Validation Perplexity was Epoch {}\n'.format(epoch, best_epoch))
                break
                
    args.epoch_last = epoch
    args.epoch_best = best_epoch

    with open(ckpt, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()
            
    with torch.no_grad():
        print('*'*55)
        print('\n')
        print('Save topics (alpha, beta)...')
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha).cpu().numpy()
        scipy.io.savemat(ckpt+'_beta.mat', {'values': beta}, do_compression=True)
        alpha = alpha.cpu().numpy()
        scipy.io.savemat(ckpt+'_alpha.mat', {'values': alpha}, do_compression=True)
        if args.train_embeddings:
            print('Save word embedding matrix (rho)...')
            rho = model.rho.weight.cpu().numpy()
            scipy.io.savemat(ckpt+'_rho.mat', {'values': rho}, do_compression=True)
        print('Compute validation perplexity...')
        val_ppl = get_completion_ppl('val', valid, rnn_inputs, model, args)
        print('Compute test perplexity...')
        test_ppl = get_completion_ppl('test', test, rnn_inputs, model, args)

    print('\n')
    print('Print and save final topics and word embeddings...')
    visualize(args, model, vocab, topic_labels, timestamps, save_output = True)
    
    print('\n')
    print('Label Topics and Save Labels to .txt...')
    topic_labels = get_topic_labels(args, alpha, embeddings, vocab)
    topic_labels_freq = get_topic_labels_by_frequency(args, alpha, embeddings, vocab)
    topic_labels_sim = get_topic_labels_by_similarity(args, alpha, embeddings, vocab)
    topic_labels_beta = get_topic_labels_from_beta(args, beta, vocab)
    all_labels = [topic_labels, topic_labels_freq, topic_labels_sim, topic_labels_beta]
    all_labels = list(map(' / '.join, zip(*all_labels)))
    
    with open(ckpt + '_topic_labels.txt','w') as f:
        for idx, k in enumerate(topic_labels):
            f.write(str(k) + '\n')

    with open(ckpt + '_all_labels.txt','w') as f:
        for idx, k in enumerate(all_labels):
            f.write(str(k) + '\n')
            
    print('\n')
    print('Print and save plots...')

    traces= {'Loss': loss_trace,
            'Val PPL': all_val_ppls,
            'NLL': nll_trace,
            'KL Alpha': kl_alpha_trace,
            'KL Eta': kl_eta_trace,
            'KL Theta': kl_theta_trace,
            'Learning Rate': lr_trace}
    try:
        test_ppl
    except NameError:
        test_ppl = ''
        
    # Plot 1 (all traces)
    fig = plt.figure(figsize=(25, 25))
    title = "{} | {}".format(args.version, execution_time)
    fig.suptitle(title)
    i = 0
    ax = []
    for key in traces:
        ax.append(fig.add_subplot(3,3, i+1))
        ax[-1].set_title(key, fontsize='x-large')
        ax[-1].set_xlabel(xlabel='epoch', horizontalalignment='right', x=1.0) #, fontstyle='italic')
        if key != 'Learning Rate':
            ax[-1].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.plot(traces[key])
        i = i+1
    try:
        fig.savefig(ckpt + '_training_plots_all.png')
    except Exception as e:
        print('...saving of plot (all) failed: ',e)
    plt.close()
    
    # Plot 2
    fig2 = plt.figure(figsize=(25, 7))
    title = "Version {} | Execution {}".format(args.version, execution_time)
    fig2.suptitle(title)
    i = 0
    ax = []
    for key in traces:
        if key in ['Learning Rate', 'Val PPL', 'Loss']:
            ax.append(fig2.add_subplot(1,3, i+1))
            ax[-1].set_title(key, fontsize='x-large')
            ax[-1].set_xlabel(xlabel='epoch', horizontalalignment='right', x=1.0) #, fontstyle='italic')
            if key != 'Learning Rate':
                ax[-1].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.plot(traces[key])
            i = i+1
    plt.show()
    try:
        fig2.savefig(ckpt + '_training_plots.png')
    except Exception as e:
        print('...saving of plot failed: ',e)
        
    print('Save settings and results to .xlsx overview file...')
    setting_exclude =  ['version', 
                        'save_path', 
                        'load_from', 
                        'ckpt',
                        'delta', 
                        'mode', 
                        'clip', 
                        'seed', 
                        'bow_norm', 
                        'num_words', 
                        'log_interval', 
                        'visualize_every', 
                        'embeddings_dim']

    cols = ['version', 'execution', 'ckpt'] + [v for v in vars(args) if v not in setting_exclude] # useful order for quick comparisons
    results = pd.DataFrame(columns = cols)
    results.loc[0, 'execution'] = execution_time
    results.loc[0, 'ckpt'] = ckpt

    for arg in vars(args):
        if arg in cols:
            results.loc[0,arg] = getattr(args, arg)

    for t in traces:
        init = traces[t][0]
        final = traces[t][-1]
        results.loc[0,t.replace(' ','_')] = final
    results.loc[0, 'val_ppl'] = val_ppl
    results.loc[0, 'test_ppl'] = test_ppl
    results.loc[0, 'TC'] = np.nan
    results.loc[0, 'TD'] = np.nan
    results.loc[0, 'TQ'] = np.nan
    results.to_excel(ckpt + '_results.xlsx')

    try: 
        previous_results = pd.read_excel(os.path.join(args.save_path, 'training_results.xlsx'), index_col=0)
        results = results.append(previous_results)
        results.reset_index(inplace=True, drop=True)
        results.to_excel(os.path.join(args.save_path, 'training_results.xlsx'))
    except Exception as ex:
        print('...saving of results into .xlsx failed:', ex)
    try:
        print('\n')
        print('Compare performance to other training and model configurations...')
        print('Version(s) with lowest val ppl:{}'.format(list(results.loc[results['val_ppl']==results['val_ppl'].min(),'version'])))
        print('Version(s) with lowest test ppl:{}'.format(list(results.loc[results['test_ppl']==results['test_ppl'].min(),'version'])))
    except Exception as e:
        print('...cannot print best version(s):',e)
        
    print('\n')
    print(10*'*')
    print('Successfully trained a D-ETM with the following settings: {}'.format(args)) 
    print('Epoch with lowest Perplexity on the Validation Set: {}'.format(args.epoch_best))
    print('Final Perplexity on the Test Set: {}'.format(test_ppl))
    print(10*'*')
    print('Execution Time:', execution_time)
    print('Checkpoint for Evaluation:', ckpt)

else: 
    print('\n')
    print('=*'*75)
    print('Evaluate a Dynamic Embedded Topic Model on Guardian Technology Data based on: {}'.format(ckpt)) 
    print('=*'*75)
    with open(ckpt, 'rb') as f:
        model = torch.load(f, map_location = device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha).cpu().numpy()
        alpha = alpha.cpu().numpy()
        
    print('\n')
    print('Load Topic Labels...')
    topic_labels = []
    with open(ckpt + '_topic_labels.txt', 'rb') as f:
        for l in f.read().splitlines():
            t_label = l.decode()
            topic_labels.append(t_label)
    
    print('\n')
    print('Visualise topics and embeddings...')
    visualize(args, model, vocab, topic_labels, timestamps, save_output=True)
    
    print('Save topic proportions (theta) and topic proportion averages for each time slice...') 
    final_theta, times_full = get_final_theta(model,args,full,rnn_inputs) 
    scipy.io.savemat(ckpt+'_theta.mat', {'values': final_theta.cpu().numpy()}, do_compression=True)
    topic_proportions = pd.DataFrame(final_theta).astype('float')
    topic_proportions.columns = ['Topic-' + str(col) for col in topic_proportions.columns]
    topic_proportions['time'] = times_full.cpu().numpy()
    topic_proportions.to_csv(ckpt+ '_theta.csv')
    topic_proportion_means = topic_proportions.groupby('time').mean()
    topic_proportion_means.to_csv(ckpt+ '_theta_avg.csv')
    
    print('\n')
    print('Compute Validation Perplexity...')
    val_ppl = get_completion_ppl('val', valid, rnn_inputs, model, args)
    print('Compute Test Perplexity...')
    test_ppl = get_completion_ppl('test', test, rnn_inputs, model, args)
    print('\n')
    print('Compute Topic Quality (Topic Coherence x Topic Diversity)...')
    quality, coherence, diversity = get_topic_quality(args,model,alpha,beta,train_tokens,vocab)
    
    print('\n')
    print('Add evaluation results to training_results.xlsx overview file...')
    results = pd.read_excel(os.path.join(args.save_path, 'training_results.xlsx'), index_col=0)
    print('\n')
    try:
        results.loc[results['version'] == args.version, 'TC'] = coherence
        results.loc[results['version'] == args.version, 'TD'] = diversity
        results.loc[results['version'] == args.version, 'TQ'] = quality  
    except Exception as e:
        print(e)
    try:
        results.to_excel(os.path.join(args.save_path, 'training_results.xlsx'))
    except Exception as ex:
        print(ex)
        
    try:
        print('\n')
        print('Version(s) with lowest val ppl:{}'.format(list(results.loc[results['val_ppl']==results['val_ppl'].dropna().min(),'version'])))
        print('Version(s) with lowest test ppl:{}'.format(list(results.loc[results['test_ppl']==results['test_ppl'].dropna().min(),'version'])))
        print('Version(s) with highest topic quality:{}'.format(list(results.loc[results['TQ']==results['TQ'].dropna().max(),'version'])))
        print('Version(s) with highest topic diversity:{}'.format(list(results.loc[results['TD']==results['TD'].dropna().max(),'version'])))
        print('Version(s) with highest topic coherence:{}'.format(list(results.loc[results['TC']==results['TC'].dropna().max(),'version'])))
    except Exception as e:
        print('Cannot print best version:',e)
        
    print('Evaluation finished.')