import numpy as np
import requests as request
import matplotlib.pyplot as plt
import networkx as nx
import cProfile as cProfile
import pstats as pstats
import pandas as pd
from sklearn.linear_model import Lasso
from scipy.optimize import minimize
from os import listdir



corpus_text = []  # documents in words
corpus = []  # documents in IDs
words = {}  # word : ID
words_inv = {}  # ID : word
K = 25  # number of topics


def ELBO(latent_param, free_param, doc):
    mu, sigma, beta = latent_param
    zeta, phi, nu, lam = free_param
    sigma_inv = np.linalg.inv(sigma)
    lam_mu_diff = (lam-mu)
    term_1 = 0.5*np.log(np.linalg.det(sigma_inv))-(K/2)*np.log(2*np.pi)\
             -(0.5*np.trace(np.dot(np.diag(nu), sigma_inv))
               +np.dot(np.dot(np.transpose(lam_mu_diff), sigma_inv), lam_mu_diff))
    term_2 = 0
    term_2_1 = -(1/zeta)*sum(np.exp(lam+nu/2))+1-np.log(zeta)
    term_3 = 0
    term_4 = 0.5*sum(np.log(nu)+np.log(2*np.pi)+1)
    term_4_1 = 0

    for n, word in enumerate(doc):
        term_2 += np.dot(lam, phi[n])
        term_3 += np.dot(phi[n], np.log(beta[:, int(word)]))
        term_4_1 += np.dot(phi[n], np.log(phi[n]))

    term_2 += len(doc)*term_2_1
    term_4 -= term_4_1
    res = term_1 + term_2 + term_3 + term_4

    return res


def opt_zeta(free_param):

    zeta, phi, nu, lam = free_param
    zeta = sum(np.exp(lam+nu/2))

    return zeta


def opt_phi(latent_param, free_param, doc):
    mu, sigma, beta = latent_param
    zeta, phi, nu, lam = free_param

    phi = np.ones((len(doc), K))
    for n, word in enumerate(doc):
        summation = 0
        for i in range(K):

            phi[n, i] = np.exp(lam[i])*beta[i, int(word)]
            summation += phi[n, i]
        phi[n] = phi[n]/summation
    return phi


def opt_lam(latent_param, free_param, doc):
    mu, sigma, beta = latent_param
    zeta, phi, nu, lam = free_param

    def elbo_df_lam(x):
        return np.dot(np.linalg.inv(sigma), (x-mu))-sum(phi)+(len(doc)/zeta)*np.exp(x+nu/2)

    def elbo_lam(x):
        return -ELBO(latent_param, (zeta, phi, nu, x), doc)

    optimized_lambda = minimize(fun=elbo_lam, x0=lam, jac=elbo_df_lam, method='Newton-CG', options={'disp': 0,'xtol': 0.00001})

    return optimized_lambda.x


def opt_nu(latent_param, free_param, doc):
    mu, sigma, beta = latent_param
    zeta, phi, nu, lam = free_param
    N = len(doc)
    sigma_inv = np.linalg.inv(sigma)
    df_1 = lambda x, i, s_inv, l: -0.5*s_inv-0.5*(N/zeta)*np.exp(l + x/2)+1/(2*x)
    df_2 = lambda x, i, l: -0.25*(N/zeta)*np.exp(l+x/2)-0.5*(1/(x*x))

    for i in range(K):
        s_inv = sigma_inv[i, i]
        l = lam[i]
        x = 5
        log_x = np.log(x)
        df1 = 1
        x_safety = 10
        while np.abs(df1) > 0.0001:
            if np.isnan(x):
                x_safety *= 2  # Two was used in the original C-code provided by the paper itself
                x = x_safety
                log_x = np.log(x)
                print("nan detected, value set to ", x)

            x = np.exp(log_x)
            df1 = df_1(x, i, s_inv, l)
            df2 = df_2(x, i, l)
            log_x -= (x*df1) / (x*x*df2+x*df1)
        nu[i] = np.exp(log_x)
    return nu


def random_free_param(doc):
    zeta = 2
    phi = np.random.dirichlet(np.ones(K), len(doc))
    nu = np.ones(K)
    lam = np.zeros(K)
    return zeta, phi, nu, lam


def init_latent_param(V):
    mu = np.zeros(K)
    sigma = np.eye(K)
    beta = np.random.dirichlet(np.ones(V), K)
    return mu, sigma, beta


def opt_doc(latent_param, doc, free_param):

    zeta, phi, nu, lam = free_param
    bound_old = ELBO(latent_param, free_param, doc)
    iters = 0

    while True:
        iters += 1
        zeta = opt_zeta(free_param)
        free_param = zeta, phi, nu, lam

        lam = opt_lam(latent_param, free_param, doc)
        free_param = zeta, phi, nu, lam

        zeta = opt_zeta(free_param)
        free_param = zeta, phi, nu, lam

        nu = opt_nu(latent_param, free_param, doc)
        free_param = zeta, phi, nu, lam

        zeta = opt_zeta(free_param)
        free_param = zeta, phi, nu, lam

        phi = opt_phi(latent_param, free_param, doc)
        free_param = zeta, phi, nu, lam


        bound = ELBO(latent_param, free_param, doc)

        if np.abs((bound_old-bound)/bound_old) > 0.00001 and iters < 500:
            bound_old = bound
        else:
            break

    return zeta, phi, nu, lam, bound_old


def expectation_step(corpus, old):

    corpus_free_params = []
    elbo_sum = 0
    for i, doc in enumerate(corpus):
        if old == 0:
            free_param = random_free_param(doc)
        else:
            zeta = 2
            phi = old[i][0]
            nu = old[i][1]
            lam = old[i][2]
            free_param = zeta, phi, nu, lam
        values = opt_doc(latent_param, doc, free_param=free_param)
        zeta, phi, nu, lam, bound_old = values
        corpus_free_params.append((phi.copy(), nu.copy(), lam.copy()))
        elbo_sum += bound_old
    return corpus_free_params, elbo_sum


def maximization_step(corpus_free_params):

    mu, sigma, beta = latent_param

    for i in range(K):
        for j, doc in enumerate(corpus):
            phi_d = corpus_free_params[j][0]
            for n, v in enumerate(doc):
                a = (phi_d[n, i])
                beta[i, int(v)] += a
        beta[i] /= np.full((1, V), sum(beta[i]))[0]

    for j in range(len(corpus)):
        mu += corpus_free_params[j][2]
    mu /= np.full((1, K), len(corpus))[0]

    for j in range(len(corpus)):
        lam_d = corpus_free_params[j][2]
        nu_d = corpus_free_params[j][1]
        lam_mu_dif = (lam_d - mu)
        sigma += np.diag(nu_d)+np.outer((lam_mu_dif), np.transpose(lam_mu_dif))
    sigma /= np.full((K, K), len(corpus))[0]

    return mu, sigma, beta


def generate_stopwords(dir):
    sw = []
    raw_stopwords = np.array(pd.read_csv(dir, header=None, sep='\s+'))
    for s in range(len(raw_stopwords)):
        sw.append(str(raw_stopwords[s])[2:-2])
    return sw


stopwords = generate_stopwords("sw.txt")


def clean_string(s):
    s = s.translate(str.maketrans('', '', "()%[]{}\'\"@.,:;?!=+/\\&`*#$£€_^'0123456789~"))
    return s


def get_cat(s):
    r = []
    c = str(s)
    count = c.count('/schemas/atom" term="')
    c = c.replace(str('/schemas/atom" term="'), "DSFGC")
    c = c.replace(str('scheme="http://arxiv.org/schemas/atom"/>'), "DSFGC")
    #print(c)
    #for i in range(count):
    c = c.split("DSFGC", 2)
    r.append((c[1]).translate(str.maketrans('', '', ' \'\"')))
    #c = c[2]
    #print(r)
    return r


def corpus_by_single_author(au, number):
    c = []
    cats = []
    #au:M+AND+au:Blei for the author David M. Blei
    url = 'http://export.arxiv.org/api/query?search_query='+au+'&start=0&max_results='+str(number)
    data = request.get(url)
    data = str(data.content)
    data = data.replace('<title>', 'XSX')
    data = data.replace('</summary>', 'XSX')
    data = data.replace('<entry>', 'DSFGC')
    data_split = data.split('DSFGC')

    #data = clean_string(data)
    for i in range(len(data_split)-1):
        cat = get_cat(data_split[i+1])
        cats.append(cat)
        data_split_part = data_split[i+1].split('XSX')
        title_abstract = str(data_split_part[1])
        title_abstract = title_abstract.replace('</title>', ' ')
        title_abstract = title_abstract.replace('<summary>', ' ')
        title_abstract = title_abstract.replace('\\n', ' ')

        #print(title_abstract)
        c.append(clean_string(title_abstract))
    cats_clean = {}
    for j, i in enumerate(cats):
        if i[0] in cats_clean:
            cats_clean[i[0]] += 1 / len(cats)
        else:
            cats_clean[i[0]] = 1 / len(cats)
    #for i in range(len(data)):
    #    if i % 2 == 0:
    #        data[i] = 0
    #    else:
    #
    #        data[i] = data[i].replace("<title>n    <summary>", " ")
    #        c.append(clean_string(data[i]))
    #print(input)
    #print(len(c))
    return c, len(c), cats_clean


def extract_xml_elements(element, s):
    r = []
    elementEnd = '</'+str(element[1:])
    c = str(s)
    count = c.count(element)
    c = c.replace(str(element), "DSFGC")
    c = c.replace(str(elementEnd), "DSFGC")

    for i in range(count):
        c = c.split("DSFGC", 2)
        r.append(c[1])
        c = c[2]
    return r

def get_authors(input, limit):
    authors = [] # ATTENTION ! CHANGED TO 'all:' FROM 'jr:'
    url = 'http://export.arxiv.org/api/query?search_query=all:'+input+'&sortBy=lastUpdatedDate&start=0&max_results='+str(limit)
    data = request.get(url)
    data = str(data.content)
    data = data.replace('<entry>','DSFGC')
    data = data.split('DSFGC')
    for i in range(len(data)-1):
        a = extract_xml_elements('<name>', data[i+1])
        authors.append(a)

    #print(authors)
    return authors

def author_to_query(input):
    au = clean_string(input)
    au = au.split()
    aq = 'au:%22' + au[0]
    for j in range(len(au) - 2):
        aq += ' ' + au[j+1]
    aq += ' ' + au[-1] + '%22'
    return aq


def corpus_by_authors():
    c = []
    meta = []
    authors = get_authors('e', 5)
    duplicates = []
    for a in range(len(authors)):

        for b in authors[a][0:3]:
            aa = b.split('\\', 1)[0]
            aa = str(aa).rsplit(' ', 1)[-1]
            if aa not in duplicates:
                try:
                    q = author_to_query(b)
                except:
                    q = 'nobody'
                res = corpus_by_single_author(str(q), 15)
                cor, count, cat = res
                if count > 1:
                    meta.append((b, count, cat))
                    c.append(cor)
                    duplicates.append(aa)
    #print(meta)
    print(len(meta))
    #np.savetxt('authors.txt', meta, fmt='%s')
    return c


docs = corpus_by_authors()
print(docs)
docs = np.concatenate(docs)
w = """
# to extract documents from a folder
def corpus_by_directory(dir, limit):
    c = []
    counter = 0  # dir = "training"
    for id in listdir(dir):
        if counter == limit:
            return c
        else:
            try:
                cor = np.array(pd.read_csv(dir+"/"+str(id), delimiter='\t'))
                d = ""
                for i in cor:
                    d += str(i)
                d = clean_string(d)
                c.append(d)
                counter += 1
            except:
                cor = ""
    return c


#docs = corpus_by_directory("training", 5000)
"""

def main():
    index = 0
    reappearance = {}
    for doc in docs:
        row_of_text = doc.split()
        row_of_text = [word for word in row_of_text if not word.lower() in stopwords]
        row_of_text = [word for word in row_of_text if not len(word) > 18 or len(word) == 0]

        row_of_IDs = np.zeros(len(row_of_text))
        for v, word in enumerate(row_of_text):
            if word.lower() in words:
                reappearance[word.lower()] += 1
            else:
                words[word.lower()] = int(index)
                words_inv[int(index)] = word.lower()
                reappearance[word.lower()] = 1
                index += 1
            row_of_IDs[v] = int(words[word.lower()])
        corpus_text.append(row_of_text)
        corpus.append(row_of_IDs)
    # unique_word_ratio = index/(index+reappearance)
    global V
    V = len(words)
    print(words)
    print(reappearance)

    global latent_param
    latent_param = init_latent_param(V)
    i = 0
    elbo_old = 1
    save_words_inv = [words_inv]
    #np.savetxt('words_inv.txt', save_words_inv, fmt='%s')
    corpus_free_params_old = 0
    while True:
        print('iter: ', i, 'elbo: ', elbo_old)
        corpus_free_params, elbo = expectation_step(corpus, corpus_free_params_old)
        convergence = (elbo_old-elbo)/elbo_old
        print('relative change: ', convergence, '\n')
        mu, sigma, beta = maximization_step(corpus_free_params)
        latent_param = mu, sigma, beta
        corpus_free_params_old = corpus_free_params
        #save numbers at every interation
        beta_full = []
        mu_full = []
        sigma_full = []
        for j in range(K):
            mu_full.append(latent_param[0][j])
            sigma_full.append(latent_param[1][j])
            beta_full.append(latent_param[2][j])
        #np.savetxt('beta_numbers.txt', beta_full, fmt='%s')
        #np.savetxt('sigma_numbers.txt', sigma_full, fmt='%s')
        #np.savetxt('mu_numbers.txt', mu_full, fmt='%s')

        lam_raw = []
        for j in range(len(corpus)):
            lam_raw.append(corpus_free_params[j][2])
        #np.savetxt("lam_raw.txt", lam_raw)
        i += 1
        if convergence > 0.0005 and i < 1000:
            elbo_old = elbo
            if elbo_old > elbo and i > 4:
                print("WARNING! - DIVERGENCE")
                break
        else:
            break
    return


main()
    #cProf = cProfile.Profile()
    #cProf.enable()
    #main()
    #cProf.disable()
    #stats = pstats.Stats(cProf).sort_stats('tottime')
    #stats.print_stats()


