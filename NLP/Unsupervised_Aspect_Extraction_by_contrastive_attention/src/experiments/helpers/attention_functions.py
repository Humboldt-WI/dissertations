# Attention functions that can be used by CAt.
# Based on the original implementation by Tulkens/Van Cranenburgh
# https://github.com/clips/cat/blob/master/cat/simple.py


import numpy as np
import sklearn.metrics.pairwise as skl_pairwise


# wrappers to absorb gamma
def linear_kernel(x, y, *args, **kwargs):
    """Wrapper around the sklearn function `linear_kernel` absorbing additional parameters."""
    return skl_pairwise.linear_kernel(x, y)


def cosine_similarity(x, y, *args, **kwargs):
    """Wrapper around the sklearn function `linear_kernel` absorbing additional parameters."""
    return skl_pairwise.cosine_similarity(x, y)


def rbf_kernel(x, y, gamma, *args, **kwargs):
    """Wrapper around the sklearn function `linear_kernel` absorbing additional parameters."""
    return skl_pairwise.rbf_kernel(x, y, gamma)


# Mean and Sum of Word Embeddings
def mowe(vec, *args, **kwargs):
    """Mean of word embeddings (MOWE) weights"""
    return sowe(vec) / len(vec)


def sowe(vec, *args, **kwargs):
    """Sum of word embeddings (SOWE) weights"""
    return np.ones(len(vec))


# Attention functions
def attention(vec, memory, gamma, similarity_function):
    """Return the attention distribution for a sentence."""
    z = similarity_function(vec, memory, gamma)
    s = z.sum()
    if s == 0:
        # If s = 0, use MOWE
        att = mowe(vec, memory)
    else:
        att = z.sum(1) / s
    return att[None, :]


def dp_attention(vec, memory, gamma):
    """Attention function using the linear kernel (dot product)."""
    return attention(vec, memory, gamma, linear_kernel)


def cos_attention(vec, memory, gamma):
    """Attention function using cosine similarity."""
    return attention(vec, memory, gamma, cosine_similarity)


def rbf_attention(vec, memory, gamma):
    """Attention function using RBF kernel"""
    return attention(vec, memory, gamma, rbf_kernel)


def no_attention(vec, memory, gamma):
    """Return MOWE weights."""
    return mowe(vec)[None, :]
