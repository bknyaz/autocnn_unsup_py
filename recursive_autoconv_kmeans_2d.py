# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 08:25:22 2016

Follow Fuel installation guidance from http://fuel.readthedocs.io/en/latest/setup.html

cd $FUEL_DATA_PATH
fuel-download cifar10
fuel-convert cifar10

K-means + recursive autoconvolution

@author: Boris Knyazev
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import theano
import theano.tensor as T

from fuel.datasets import H5PYDataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import image
from skimage.transform import resize
from scipy.linalg import svd
from sklearn.preprocessing import normalize

def autoconv2d(X):
    sz = X.shape # X - 4D array n x K x h x w
    X -= np.reshape(np.mean(X, axis=(2,3)),(-1,sz[-3],1,1)) # local centering
    X /= np.reshape(np.std(X, axis=(2,3)),(-1,sz[1],1,1)) + 1e-10 # local scaling
    X = np.pad(X, ((0,0),(0,0),(0,sz[-2]-1),(0,sz[-1]-1)), 'constant') # zero-padding to compute linear convolution
    return np.real(np.fft.ifft2(np.fft.fft2(X)**2))

def resize_batch(X, new_size):
    out = []
    for x in X:
        out.append(resize(x.transpose((1,2,0)), new_size, order=1, preserve_range=True).transpose((2,0,1)))
    return np.stack(out)

def mat2gray(X):
    m = np.min(X,axis=1,keepdims=True)
    X_range = np.max(X,axis=1,keepdims=True)-m
    idx = np.squeeze(X_range == 0)
    X[idx,:] = 0
    X[np.logical_not(idx),:] = (X[np.logical_not(idx),:]-m[np.logical_not(idx)])/X_range[np.logical_not(idx)]
    return X

# https://github.com/sdanaipat/Theano-ZCA
class ZCA(object):
    def __init__(self):
        X_in = T.matrix('X_in')
        u = T.matrix('u')
        s = T.vector('s')
        eps = T.scalar('eps')

        X_ = X_in - T.mean(X_in, 0)
        sigma = T.dot(X_.T, X_) / X_.shape[0]
        self.sigma = theano.function([X_in], sigma)

        Z = T.dot(T.dot(u, T.nlinalg.diag(1. / T.sqrt(s + eps))), u.T)
        X_zca = T.dot(X_, Z.T)
        self.compute_zca = theano.function([X_in, u, s, eps], X_zca)

        self._u = None
        self._s = None

    def fit(self, X):
        cov = self.sigma(X)
        u, s, _ = svd(cov)
        self._u = u.astype(np.float32)
        self._s = s.astype(np.float32)
        del cov

    def transform(self, X, eps):
        return self.compute_zca(X, self._u, self._s, eps)


    def fit_transform(self, X, eps):
        self.fit(X)
        return self.transform(X, eps)

def MiniBatchKMeansAutoConv(X, patch_size, n_clusters, conv_orders, batch_size=100):
    # data: n x K x h x w
    sz = X.shape
    X = image.PatchExtractor(patch_size=patch_size,max_patches=max(1,np.round(5/max(1,len(conv_orders))))).transform(X.transpose((0,2,3,1))) # n x h x w x K
    X = np.asarray(X,dtype='float32')/255
    X = X-np.reshape(np.mean(X, axis=(1,2)),(-1,1,1,3)) # local centering
    X = X/(np.reshape(np.std(X, axis=(1,2)),(-1,1,1,3)) + 1e-10) # local scaling
    X = X.transpose((0,3,1,2)) # -> n x K x h x w
    n_batches = int(np.ceil(len(X)/float(batch_size)))
    autoconv_patches = []
    for batch in range(n_batches):
        X_order = np.squeeze(X[np.arange(batch*batch_size, min(len(X)-1,(batch+1)*batch_size))])
        # Recursive autoconvolution
        for conv_order in conv_orders:
            if conv_order > 0:
                X_order = autoconv2d(X_order)
                # resize by a factor of 2
                if np.random.rand() > 0.5:
                    s = (X_order.shape[2]-1)/4
                    X_sampled = X_order[:,:,s:-s,s:-s]
                else:
                    X_sampled = resize_batch(X_order, [int(np.round(s/2.)) for s in X_order.shape[2:]])
                if conv_order > 1:
                    X_order = X_sampled
                if X_sampled.shape[2] != X.shape[2]:
                    X_sampled = resize_batch(X_sampled, X.shape[2:])
            else:
                X_sampled = X_order
            autoconv_patches.append(X_sampled)
        print('%d/%d ' % (batch,n_batches))
    X = np.concatenate(autoconv_patches) # n x K x h x w
    X = np.asarray(X.reshape(X.shape[0],-1),dtype='float32')
    X = mat2gray(X)
    X = ZCA().fit_transform(X, 1e-5)
    X = normalize(X)
    km = MiniBatchKMeans(n_clusters = n_clusters, batch_size=10*n_clusters).fit(X).cluster_centers_
    return km.reshape(-1,sz[1],patch_size[0],patch_size[1])

# Visualization of filters
# http://sklearn-theano.github.io/auto_examples/plot_overfeat_layer1_filters.html
def make_visual(layer_weights):
    max_scale = layer_weights.max(axis=-1).max(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    min_scale = layer_weights.min(axis=-1).min(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    return (255 * (layer_weights - min_scale) /
            (max_scale - min_scale)).astype('uint8')
                
def make_mosaic(layer_weights, sz):
    # Dirty hack (TM)
    lw_shape = layer_weights.shape
    lw = make_visual(layer_weights).reshape(sz[0], sz[1], *lw_shape[1:])
    lw = lw.transpose(0, 3, 1, 4, 2)
    lw = lw.reshape(sz[0] * lw_shape[-1], sz[1] * lw_shape[-2], lw_shape[1])
    return lw


def plot_filters(layer_weights, sz=(8,9), title=None, show=False):
    mosaic = make_mosaic(layer_weights, sz)
    plt.imshow(mosaic, interpolation='nearest')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
        plt.imsave('filters.png', mosaic)


# Example

# load CIFAR-10 data
fuel_path = os.getenv('FUEL_DATA_PATH')
train_set = H5PYDataset(fuel_path + '/cifar10.hdf5', which_sets=('train',))
handle = train_set.open()
n_samples = 10000
data = train_set.get_data(handle, slice(0, n_samples)) 
train_set.close(handle)
plt.imshow(np.transpose(data[0][0,:,:,:],(1,2,0)))
plt.show()

# run k-meanss
km = MiniBatchKMeansAutoConv(data[0], (11,11), 128, range(4))
plot_filters(km, sz=(8,16), title='Learned filters', show=True)