"""This file contains some (adapted) code obtained from https://github.com/adjidieng/DETM, 10/2020.
It includes functionality to obtain data, visualise results, find neighbours in the embedding space,
label topics, obtain/save results and to evaluate performance"""

import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool
from collections import Counter
import math 
import matplotlib.pyplot as plt 
import matplotlib 
import numpy as np
import pandas as pd
import os
import pickle
import pytz
import random
import scipy.io
from sklearn.manifold import TSNE
import sys
import torch 
from torch import nn, optim
from torch.nn import functional as F

tiny = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Data ##

def fetch_data(path, name):
    if name == 'full':
        token_file = os.path.join(path, 'bow_full_tokens.mat')
        count_file = os.path.join(path, 'bow_full_counts.mat')
        time_file = os.path.join(path, 'bow_full_timestamps.mat')
    elif name == 'train':
        token_file = os.path.join(path, 'bow_tr_tokens.mat')
        count_file = os.path.join(path, 'bow_tr_counts.mat')
        time_file = os.path.join(path, 'bow_tr_timestamps.mat')
    elif name == 'valid':
        token_file = os.path.join(path, 'bow_va_tokens.mat')
        count_file = os.path.join(path, 'bow_va_counts.mat')
        time_file = os.path.join(path, 'bow_va_timestamps.mat')
    else:
        token_file = os.path.join(path, 'bow_ts_tokens.mat')
        count_file = os.path.join(path, 'bow_ts_counts.mat')
        time_file = os.path.join(path, 'bow_ts_timestamps.mat')
    tokens = scipy.io.loadmat(token_file)['tokens'].squeeze()
    counts = scipy.io.loadmat(count_file)['counts'].squeeze()
    times = scipy.io.loadmat(time_file)['timestamps'].squeeze()
    if name == 'test':
        token_1_file = os.path.join(path, 'bow_ts_h1_tokens.mat')
        count_1_file = os.path.join(path, 'bow_ts_h1_counts.mat')
        token_2_file = os.path.join(path, 'bow_ts_h2_tokens.mat')
        count_2_file = os.path.join(path, 'bow_ts_h2_counts.mat')
        tokens_1 = scipy.io.loadmat(token_1_file)['tokens'].squeeze()
        counts_1 = scipy.io.loadmat(count_1_file)['counts'].squeeze()
        tokens_2 = scipy.io.loadmat(token_2_file)['tokens'].squeeze()
        counts_2 = scipy.io.loadmat(count_2_file)['counts'].squeeze()
        return {'tokens': tokens, 'counts': counts, 'times': times, 
                    'tokens_1': tokens_1, 'counts_1': counts_1, 
                        'tokens_2': tokens_2, 'counts_2': counts_2} 
    return {'tokens': tokens, 'counts': counts, 'times': times}

def get_data(path):
    with open(os.path.join(path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    full = fetch_data(path, 'full')
    train = fetch_data(path, 'train')
    valid = fetch_data(path, 'valid')
    test = fetch_data(path, 'test')
    return vocab, full, train, valid, test

def get_batch(tokens, counts, ind, vocab_size, emsize=300, times=None):
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    times_batch = np.zeros((batch_size, ))
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        timestamp = times[doc_id]
        times_batch[i] = timestamp
        L = count.shape[1]
        if len(doc) == 1: 
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    times_batch = torch.from_numpy(times_batch).to(device)
    return data_batch, times_batch
    return data_batch

def get_rnn_input(tokens, counts, times, num_times, vocab_size, num_docs):
    indices = torch.randperm(num_docs)
    indices = torch.split(indices, 1000) 
    rnn_input = torch.zeros(num_times, vocab_size).to(device)
    cnt = torch.zeros(num_times, ).to(device)
    for idx, ind in enumerate(indices):
        data_batch, times_batch = get_batch(tokens, counts, ind, vocab_size, times=times)
        for t in range(num_times):
            tmp = (times_batch == t).nonzero()
            docs = data_batch[tmp].squeeze().sum(0)
            rnn_input[t] += docs
            cnt[t] += len(tmp)
    rnn_input = rnn_input / cnt.unsqueeze(1)
    return rnn_input


## Visualisation / Prints ##

def visualize(args, model, vocab, topic_labels, timestamps, save_output = False):
    """Visualises topics, word usage evolution and nearest neighbours for selected words."""
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        print('Visualise topics...')
        times = list(range(args.num_times)) 
        for k in range(args.num_topics):
            if topic_labels is not None:
                print('\n')
                print('Topic {}: "{}"'.format(k, topic_labels[k]))
            else:
                print('\n')
                print('Topic {} '.format(k))

            for t in times:
                gamma = beta[k, t, :]
                top_words = list(gamma.cpu().numpy().argsort()[-args.num_words:][::-1])
                topic_words = [vocab[a] for a in top_words]
                
                if args.mode == 'eval':
                    print('...Time {}: {}'.format(timestamps[t], topic_words))
                else:
                    if t==0 or t%4==0:
                        print('...Time {}: {}'.format(timestamps[t], topic_words))
                
                if save_output:
                    try:
                        if k == 0 and t == 0:
                            f = open(args.ckpt + '_topic_nn_prints.txt', 'w+')
                            f.write('\nTopic {} .. Time: {} ===> {}\n'.format(k, timestamps[t], topic_words))
                            f.close()      
                        else:
                            f = open(args.ckpt + '_topic_nn_prints.txt', 'a')
                            f.write('Topic {} .. Time: {} ===> {}\n'.format(k, timestamps[t], topic_words))
                            f.close()
                    except Exception as n:
                        continue
        print('\n')
        print('Visualise word embeddings ...')
        queries = ['Spotify','social_media', '3D', 'security', 'PS4', '5G', 'A.I.','AI', 'artificial_intelligence', 'Mark_Zuckerberg', 'Apple']
        try:
            embeddings = model.rho.weight
        except:
            embeddings = model.rho
        neighbors = []
        for word in queries:
            try:
                print('"{}"" .. neighbours: {}'.format(
                    word, nearest_neighbors(word, embeddings, vocab, args.num_words+1)[-args.num_words:]))
            except:
                continue
            if save_output:
                try:
                    f = open(args.ckpt + '_topic_nn_prints.txt', 'a')
                    f.write('\nword: {} .. neighbours: {}'.format(word, nearest_neighbors(word, embeddings, vocab, args.num_words+1)[-args.num_words:]))
                    f.close() 
                except Exception as x:
                    continue
                    print('...failed to save word embeddings to .txt file: ',x) 
        print('-'*100)

    
## Results ##

def eta_helper(rnn_inp, model):
    inp = model.q_eta_map(rnn_inp).unsqueeze(1)
    hidden = model.init_hidden()
    output, _ = model.q_eta(inp, hidden)
    output = output.squeeze()
    etas = torch.zeros(model.num_times, model.num_topics).to(device)
    inp_0 = torch.cat([output[0], torch.zeros(model.num_topics,).to(device)], dim=0)
    etas[0] = model.mu_q_eta(inp_0)
    for t in range(1, model.num_times):
        inp_t = torch.cat([output[t], etas[t-1]], dim=0)
        etas[t] = model.mu_q_eta(inp_t)
    return etas

def get_eta(source, rnn_inputs, model):
    model.eval()
    with torch.no_grad():
        if source in ['val','full']:
            rnn_inp = rnn_inputs[source]
            return eta_helper(rnn_inp, model)
        else:
            rnn_1_inp = rnn_inputs['test1']
            return eta_helper(rnn_1_inp, model)

def get_theta(eta, bows, model):
    model.eval()
    with torch.no_grad():
        inp = torch.cat([bows, eta], dim=1)
        q_theta = model.q_theta(inp)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta    

def get_final_theta(model,args,full,rnn_inputs):
    model.eval()
    with torch.no_grad():
        indices = torch.split(torch.tensor(range(args.num_docs_full)), args.batch_size)
        tokens = full['tokens'] #full_tokens #
        counts = full['counts'] #full_counts #
        times = full['times'] #full_times #
        eta = get_eta(source='full', rnn_inputs=rnn_inputs, model=model) 
        for idx, ind in enumerate(indices):
            data_batch, times_batch = get_batch(tokens, counts, ind, args.vocab_size, args.rho_size, times=times)
            sums = data_batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            eta_td = eta[times_batch.type('torch.LongTensor')]
            theta = get_theta(eta_td, normalized_data_batch, model)
            if idx > 0:
                theta_full = torch.cat((theta_full, theta), 0)
                times_full = torch.cat((times_full, times_batch), 0)
            else:
                theta_full = theta
                times_full = times_batch
    return theta_full, times_full

    
## Evaluation ##

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if len(doc) == 1: 
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if len(doc) == 1: 
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data, vocab):
    D = len(data) # number of documents
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                tmp += f_wi_wj
                j += 1
                counter += 1
            TC_k += tmp
        TC.append(TC_k)
    TC = np.mean(TC) / counter
    return TC, counter

def get_completion_ppl(source, data, rnn_inputs, model, args):
    """Computes document completion perplexity."""
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        
        if source == 'val':
            indices = torch.split(torch.tensor(range(args.num_docs_valid)), args.batch_size)
            tokens = data['tokens']
            counts = data['counts']
            times = data['times']
            eta = get_eta('val', rnn_inputs, model=model)
            acc_loss = 0
            cnt = 0
            for idx, ind in enumerate(indices):
                data_batch, times_batch = get_batch(tokens, counts, ind, args.vocab_size, args.rho_size, times)
                sums = data_batch.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch = data_batch / sums
                else:
                    normalized_data_batch = data_batch
                eta_td = eta[times_batch.type('torch.LongTensor')]
                theta = get_theta(eta_td, normalized_data_batch, model)
                alpha_td = alpha[:, times_batch.type('torch.LongTensor'), :]
                beta = model.get_beta(alpha_td).permute(1, 0, 2)
                loglik = theta.unsqueeze(2) * beta
                loglik = loglik.sum(1)
                loglik = torch.log(loglik)
                nll = -loglik * data_batch
                nll = nll.sum(-1)
                loss = nll / sums.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_all = round(math.exp(cur_loss), 1)
            print('{} Perplexity: {}'.format(source[0].upper()+source[1:], ppl_all))
            return ppl_all
    
        elif source == 'test': 
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.batch_size)
            times = data['times']
            tokens_1 = data['tokens_1']
            counts_1 = data['counts_1']
            tokens_2 = data['tokens_2']
            counts_2 = data['counts_2']
            eta_1 = get_eta('test', rnn_inputs, model=model)
            acc_loss = 0
            cnt = 0
            indices = torch.split(torch.tensor(range(args.num_docs_test)), args.batch_size)
            for idx, ind in enumerate(indices):
                data_batch_1, times_batch_1 = get_batch(
                    tokens_1, counts_1, ind, args.vocab_size, args.rho_size, times)
                sums_1 = data_batch_1.sum(1).unsqueeze(1)
                if args.bow_norm:
                    normalized_data_batch_1 = data_batch_1 / sums_1
                else:
                    normalized_data_batch_1 = data_batch_1
                eta_td_1 = eta_1[times_batch_1.type('torch.LongTensor')]
                theta = get_theta(eta_td_1, normalized_data_batch_1, model)
                data_batch_2, times_batch_2 = get_batch(
                    tokens_2, counts_2, ind, args.vocab_size, args.rho_size, times)
                sums_2 = data_batch_2.sum(1).unsqueeze(1)
                alpha_td = alpha[:, times_batch_2.type('torch.LongTensor'), :]
                beta = model.get_beta(alpha_td).permute(1, 0, 2)
                loglik = theta.unsqueeze(2) * beta
                loglik = loglik.sum(1)
                loglik = torch.log(loglik)
                nll = -loglik * data_batch_2
                nll = nll.sum(-1)
                loss = nll / sums_2.squeeze()
                loss = loss.mean().item()
                acc_loss += loss
                cnt += 1
            cur_loss = acc_loss / cnt
            ppl_dc = round(math.exp(cur_loss), 1)
            print('{} Doc Completion Perplexity: {}'.format(source[0].upper()+source[1:], ppl_dc))
            return ppl_dc
        else:
            return ''

def diversity_helper(beta, num_tops, args):
    list_w = np.zeros((args.num_topics, num_tops))
    for k in range(args.num_topics):
        gamma = beta[k,:]
        top_words = gamma.cpu().numpy().argsort()[-num_tops:][::-1]
        list_w[k,:] = top_words
    list_w = np.reshape(list_w, (-1))
    list_w = list(list_w)
    n_unique = len(np.unique(list_w))
    diversity = n_unique / (args.num_topics * num_tops)
    return diversity

def get_topic_quality(args,model,alpha,beta,train_tokens,vocab):
    """Computes topic coherence and topic diversity (averages over time)."""
    model.eval()
    with torch.no_grad():
        alpha = model.mu_q_alpha
        beta = model.get_beta(alpha) 
        print('Get Topic Diversity...')
        TD_all = np.zeros((args.num_times,))
        for tt in range(args.num_times): 
            TD_all[tt] = diversity_helper(beta=beta[:, tt, :], num_tops=25, args=args)
        diversity = np.mean(TD_all)
        print('Topic Diversity for each Time Slice:', TD_all)
        print('Average Topic Diversity (TD): {}'.format(diversity))
        print('Get Topic Coherence...')
        TC_all = []
        cnt_all = []
        for tt in range(args.num_times):
            tc, cnt = get_topic_coherence(beta[:, tt, :].cpu().numpy(), train_tokens, vocab)
            print('Average Topic Coherence for Time Slice {}: {}'.format(tt,tc))
            TC_all.append(tc)
            cnt_all.append(cnt)
        coherence = np.mean(TC_all)
        print('Final (Average) Topic Coherence (TC):', coherence)
        quality = coherence * diversity
        print('Topic Quality: {} (TC {} x TD {})'.format(quality, coherence, diversity))
        print('#'*100)
        return quality, coherence, diversity
        
## Neighbours ##

def nearest_neighbors(word, embeddings, vocab, num_words):
    vectors = embeddings.cpu().numpy() 
    index = vocab.index(word)
    query = embeddings[index].cpu().numpy() 
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:num_words]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors

def nearest_neighbors_from_vector(vector, embeddings, vocab, num_words):
    embeddings = embeddings.cpu().numpy() 
    ranks = embeddings.dot(vector).squeeze()
    denom = vector.T.dot(vector).squeeze()
    denom = denom * np.sum(embeddings**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:num_words]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors

def cosine_neighbors(vector, embeddings, vocab):
    embeddings = embeddings.cpu().numpy() 
    numerator = embeddings.dot(vector).squeeze()
    denominator = vector.T.dot(vector).squeeze()
    denominator = denominator * np.sum(embeddings**2, 1)
    similarity = numerator / np.sqrt(denominator)
    distances = 1 - similarity
    return similarity, distances


## Topic Labeling ##

def get_topic_labels(args, alpha, embeddings, vocab):
    topic_labels = []
    topic_labels_dupl = {}
    for k in range(args.num_topics):
        label = []
        for t in range(args.num_times):
            if t > 0:
                sim, distances = cosine_neighbors(alpha[k, t, :], embeddings, vocab)#, 1)
                distances_avg = np.append(distances_avg, [distances], axis=0)
            else:
                sim, distances = cosine_neighbors(alpha[k, t, :], embeddings, vocab)#, 1)
                distances_avg = np.array([distances])
        distances_avg = distances_avg.mean(axis=0)
        distances_avg_sorted = list(np.sort(distances_avg))
        distances_idx = list(np.argsort(distances_avg))
        minimum = distances_avg_sorted[0]
        threshold = 0.01
        for idx, d in enumerate(distances_avg_sorted):
            candidate_label = vocab[distances_idx[idx]]
            add_crit = not any(candidate_label.lower() in l.lower() for l in label) and not any(l.lower() in candidate_label.lower() for l in label)
            if d <= (1+threshold)*minimum and add_crit and len(label)<2:
                label.append(candidate_label)
        label_str = ", " .join(label[:-1]) + ' & ' + label[-1] if len(label)>1 else str(label[0])
        label_duplicate_count = topic_labels.count(label_str)
        if label_duplicate_count > 0:
            topic_labels.append(label_str + ' ' + str(label_duplicate_count+1))
        else:
            topic_labels.append(label_str)
    return topic_labels

def get_topic_labels_by_similarity(args, alpha, embeddings, vocab):
    topic_labels = []
    topic_labels_dupl = {}
    for k in range(args.num_topics):
        label = []
        for t in range(args.num_times):
            if t > 0:
                sim, distances = cosine_neighbors(alpha[k, t, :], embeddings, vocab)
                sim_avg = np.append(sim_avg, [sim], axis=0)
            else:
                sim, distances = cosine_neighbors(alpha[k, t, :], embeddings, vocab)
                sim_avg = np.array([sim])
        sim_avg = sim_avg.mean(axis=0)
        sim_avg_sorted = list(np.sort(sim_avg))[::-1]
        sim_avg_idx = list(np.argsort(sim_avg))[::-1]
        maximum = sim_avg_sorted[0]
        threshold = 1e-10
        for idx, d in enumerate(sim_avg_sorted):
            candidate_label = vocab[sim_avg_idx[idx]]
            add_crit = not any(candidate_label.lower() in l.lower() for l in label) and not any(l.lower() in candidate_label.lower() for l in label)
            cut_off = (1+threshold)*maximum if maximum<0 else (1-threshold)*maximum
            if d <= cut_off and add_crit and len(label)<2:
                label.append(candidate_label)
        label_str = ", " .join(label[:-1]) + ' & ' + label[-1] if len(label)>1 else str(label[0])
        label_duplicate_count = topic_labels.count(label_str)
        if label_duplicate_count > 0:
            topic_labels.append(label_str + ' ' + str(label_duplicate_count+1))
        else:
            topic_labels.append(label_str)
    return topic_labels

def get_topic_labels_by_frequency(args, alpha, embeddings, vocab):
    topic_labels = []
    for k in range(args.num_topics):
        label = []
        nearest = []
        for t in range(args.num_times):
            nn = nearest_neighbors_from_vector(alpha[k,t,:], embeddings, vocab, 1)[0]
            nearest.append(nn)
        counts = dict(Counter(nearest))
        maximum = max(counts.values())
        for key, value in counts.items():
            add_crit = not any(key.lower() in l.lower() for l in label) and not any(l.lower() in key.lower() for l in label)
            if value == maximum and add_crit:
                label.append(key)
        topic_labels.append(' & '.join(label))
    return topic_labels

def get_topic_labels_from_beta(args, beta, vocab):
    beta_avg = np.mean(beta,axis=1)
    topic_labels = []
    for k in range(args.num_topics):
        top_words_avg = beta_avg[k,:]
        values = list(np.sort(top_words_avg)[::-1])
        indices = list(np.argsort(top_words_avg)[::-1])
        top = values[0]
        top_label = vocab[indices[0]]
        second = values[1]
        second_label = vocab[indices[1]] 
        threshold = 0.9*top
        if second>threshold and not ((top_label in second_label) or (second_label in top_label)):
            label = '{} & {}'.format(top_label,second_label)
        else:
            label = top_label
        topic_labels.append(label)
    return topic_labels