# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:00:00 2018

@author: Abhishek S
"""

from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import copy
from io import BytesIO
import sframe as sf



?multivariate_normal.pdf
multivariate_normal.pdf(1,mean=0,cov=1)

print multivariate_normal.pdf([10,5],mean=[3,4],cov=3)
print norm.pdf(3)

data_pts=pd.DataFrame({'X':[10,2,3],'Y':[5,1,7]})
data_pts=np.array(data_pts)
clusters=np.array([[3,4],[6,3],[4,6]])

dist=pairwise_distances(data_pts,clusters,metric='euclidean')
?np.argmin
np.argmin(dist,axis=1)
l=list()
for i in clusters:
    l.append(multivariate_normal.pdf(data_pts,mean=i,cov=[[3,0],[0,3]]))

?np.array
l=np.array(l,shape=[3,3])
c_wts=[1./3,1./3,1./3]
l=l*c_wts
l=l.T
?normalize
res=normalize(l,norm='l1',axis=1)
c_wts_1=np.sum(res,axis=0)/np.sum(res)

a=np.array([[1,1],[2,2]])
b=np.array([3,4])
a*b
b[:,np.newaxis]
?np.newaxis
a*b.reshape([2,1])
res[:,0]
np.sum(data_pts*res[:,0][:,np.newaxis],axis=0)/np.sum(res[:,0])

cluster=np.dot(data_pts.T,res).T/np.sum(res,axis=0).reshape([3,1])

diff=data_pts-cluster[0,:]
np.outer(diff[0,:],diff[0,:])


def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll


data_pts
clusters[0,:]
multivariate_normal.pdf(data_pts,mean=clusters[0,:],cov=3)

a=np.zeros([3,3])
a[:,0]=multivariate_normal.pdf(data_pts,mean=clusters[0,:],cov=3)
a
b
c=a*b[:,np.newaxis]
normalize(c,norm='l1',axis=1)
def compute_responsibilities(data,weights,means,covariances):
    num_data=len(data)
    num_cluster=len(weights)
    rik=np.zeros([num_data,num_cluster])
    for i in xrange(num_cluster):
        rik[:,i]=multivariate_normal.pdf(data,mean=means[i],cov=covariances[i])
    rik=rik*weights
    rik=normalize(rik,norm='l1',axis=1)
    return rik

resp = compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])

if resp.shape==(2,2) and np.allclose(resp, np.array([[0.10512733, 0.89487267], [0.46468164, 0.53531836]])):
    print 'Checkpoint passed!'
else:
    print 'Check your code again.'

def compute_soft_counts(resp):
    return np.sum(resp,axis=0)
compute_soft_counts(resp)

def compute_weights(counts):
    return(normalize(counts.reshape(1,-1),norm='l1').flatten())

resp = compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.],[0,0]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = compute_soft_counts(resp)
weights = compute_weights(counts)

print counts
print weights

if np.allclose(weights, [0.27904865942515705, 0.720951340574843]):
    print 'Checkpoint passed!'
else:
    print 'Check your code again.'
    
def compute_means(data,resp,counts):
    num_cluster=len(resp)
    dt=data.copy()
    mn=np.dot(data.T,resp)
    mn=mn/counts
    return mn.T

data=data_tmp

data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = compute_soft_counts(resp)
means = compute_means(data_tmp, resp, counts)

if np.allclose(means, np.array([[-0.6310085, -1.262017], [0.25140299, 0.50280599]])):
    print 'Checkpoint passed!'
else:
    print 'Check your code again.'

def s_outer(a):
    return(np.outer(a,a))

i=0
10.64075489*0.105

def compute_covariances(data,resp,counts,means):
    num_cluster=len(counts)
    num_data=len(data)
    vr=list()
    for i in xrange(num_cluster):
        dt=data-means[i,:]
        dt=np.apply_along_axis(s_outer,1,dt)
        dt=dt*resp[:,i][:,np.newaxis][:,np.newaxis]
        wt=0
        for j in dt:
            wt+=j
        wt=wt/counts[i]
        vr.append(wt)
    return(vr)
data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = compute_soft_counts(resp)
means = compute_means(data_tmp, resp, counts)
covariances = compute_covariances(data_tmp, resp, counts, means)

if np.allclose(covariances[0], np.array([[0.60182827, 1.20365655], [1.20365655, 2.4073131]])) and \
    np.allclose(covariances[1], np.array([[ 0.93679654, 1.87359307], [1.87359307, 3.74718614]])):
    print 'Checkpoint passed!'
else:
    print 'Check your code again.'

def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    means=init_means[:]
    covar=init_covariances[:]
    wts=init_weights[:]
    ll=loglikelihood(data,wts,means,covar)
    ll_trace=[ll]
    for i in xrange(maxiter):
        if i%5==0:
            print "Iteration : %s"%(i)
        resp=compute_responsibilities(data,wts,means,covar)
        counts=compute_soft_counts(resp)
        wts=compute_weights(counts)
        means=compute_means(data,resp,counts)
        covar=compute_covariances(data,resp,counts,means)
        ll_new=loglikelihood(data,wts,means,covar)
        ll_trace.append(ll_new)
        if (ll_new-ll)<thresh and ll_new>-np.Inf:
            break
        ll=ll_new
    print "Iteration :",i
    out={'weights':wts,'means':means,'loglik':ll_trace,'covs':covar,'resp':resp}
    return out

def generate_MoG_data(num_data,means,covariances,weights):
    num_cluster=len(weights)
    data=[]
    for i in xrange(num_data):
        k=np.random.choice(num_cluster,1,p=weights)[0]
        x=np.random.multivariate_normal(means[k],covariances[k])
        data.append(x)
    return data

init_means=[
        [5,0],
        [1,1],
        [0,5]
        ]        
init_covariances=[
        [[.5,0],[0,.5]],
        [[.92, .38], [.38, .91]], # covariance of cluster 2
        [[.5, 0.], [0, .5]] 
        ]
init_weights = [1/4., 1/2., 1/4.]  # weights of each cluster

# Generate data
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covariances, init_weights)

plt.figure()
d = np.vstack(data)
plt.plot(d[:,0], d[:,1],'ko')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

np.random.seed(4)
# Initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_weights = [1/3.] * 3

?np.cov

# Run EM 
data=d
results = EM(data, initial_means, initial_covs, initial_weights)
results['weights']
results['means']
results['covs']



import matplotlib.mlab as mlab



def plot_contours(data, means, covs, title):
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors = col[i])
        plt.title(title)
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
# Parameters after initialization
plot_contours(data, initial_means, initial_covs, 'Initial clusters')

# Parameters after running EM to convergence
results = EM(data, initial_means, initial_covs, initial_weights)
plot_contours(data, results['means'], results['covs'], 'Final clusters')

# YOUR CODE HERE
results =EM(data,initial_means,initial_covs,initial_weights,maxiter=12)

plot_contours(data, results['means'], results['covs'], 'Clusters after 12 iterations')

loglikelihoods=results['loglik']
plt.plot(range(len(loglikelihoods)), loglikelihoods, linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size':16})
plt.tight_layout()


images = sf.SFrame('E:/Clustering and Retreival/Week 4/images.sf')
import array
images['rgb'] = images.pack_columns(['red', 'green', 'blue'])['X4']

np.random.seed(1)

# Initalize parameters
init_means = [images['rgb'][x] for x in np.random.choice(len(images), 4, replace=False)]
cov = np.diag([images['red'].var(), images['green'].var(), images['blue'].var()])
init_covariances = [cov, cov, cov, cov]
init_weights = [1/4., 1/4., 1/4., 1/4.]

# Convert rgb data to numpy arrays
img_data = [np.array(i) for i in images['rgb']]  
img_data=np.vstack(img_data)
# Run our EM algorithm on the image data using the above initializations. 
# This should converge in about 125 iterations
out = EM(img_data, init_means, init_covariances, init_weights)

ll = out['loglik']
plt.plot(range(len(ll)),ll,linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size':16})
plt.tight_layout()

plt.figure()
plt.plot(range(3,len(ll)),ll[3:],linewidth=4)
plt.xlabel('Iteration')
plt.ylabel('Log-likelihood')
plt.rcParams.update({'font.size':16})
plt.tight_layout()


import colorsys
def plot_responsibilities_in_RB(img, resp, title):
    N, K = resp.shape
    
    HSV_tuples = [(x*1.0/K, 0.5, 0.9) for x in range(K)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    
    R = img['red']
    B = img['blue']
    resp_by_img_int = [[resp[n][k] for k in range(K)] for n in range(N)]
    cols = [tuple(np.dot(resp_by_img_int[n], np.array(RGB_tuples))) for n in range(N)]

    plt.figure()
    for n in range(len(R)):
        plt.plot(R[n], B[n], 'o', c=cols[n])
    plt.title(title)
    plt.xlabel('R value')
    plt.ylabel('B value')
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()
N, K = out['resp'].shape
random_resp = np.random.dirichlet(np.ones(K), N)
plot_responsibilities_in_RB(images, random_resp, 'Random responsibilities')

out = EM(img_data, init_means, init_covariances, init_weights, maxiter=1)
plot_responsibilities_in_RB(images, out['resp'], 'After 1 iteration')

out = EM(img_data, init_means, init_covariances, init_weights, maxiter=20)
plot_responsibilities_in_RB(images, out['resp'], 'After 20 iterations')

out['means']
out['covs']

for i in range(len(out['means'])):
    print multivariate_normal.pdf(img_data[0,:],mean=out['means'][i],cov=out['covs'][i])*out['weights'][i]

weights=out['weights']
covs=out['covs']
means=out['means']
resp=compute_responsibilities(img_data,weights,means,covs)


weights = out['weights']
means = out['means']
covariances = out['covs']
rgb = images['rgb']
N = len(images) # number of images
K = len(means) # number of clusters

assignments = [0]*N
probs = [0]*N

for i in range(N):
    # Compute the score of data point i under each Gaussian component:
    p = np.zeros(K)
    for k in range(K):
        p[k] = weights[k]*multivariate_normal.pdf(rgb[i], mean=means[k], cov=covariances[k])
        
    # Compute assignments of each data point to a given cluster based on the above scores:
    assignments[i] = np.argmax(p)
    
    # For data point i, store the corresponding score under this cluster assignment:
    probs[i] = np.max(p)

assignments = sf.SFrame({'assignments':assignments, 'probs':probs, 'image': images['image']})

def get_top_images(assignments, cluster, k=5):
    # YOUR CODE HERE
    images_in_cluster = assignments[assignments['assignments']==cluster]
    top_images = images_in_cluster.topk('probs', k)
    return top_images['image']

def save_images(images, prefix):
    for i, image in enumerate(images):
        Image.open(BytesIO(image._image_data)).save(prefix % i)

for component_id in range(4):
    print 'Component {0:d}'.format(component_id)
    images = get_top_images(assignments, component_id)
#    save_images(images)
    save_images(images, 'component_{0:d}_%d.jpg'.format(component_id))
    print '\n'
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import os
cwd = os.getcwd()
#=============================================================================================================================#

import sframe                                            # see below for install instruction
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.stats import multivariate_normal
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import pandas as pd
import json

wiki=pd.read_csv('E:/Clustering and Retreival/Week 4/people_wiki.csv').head(5000)
wiki.head()

def load_sparse_data(filename):
    loader=np.load(filename)
    data=loader['data']
    indptr=loader['indptr']
    indices=loader['indices']
    shape=loader['shape']
    return csr_matrix((data,indices,indptr),shape)

tf_idf=load_sparse_data('E:/Clustering and Retreival/Week 4/4_tf_idf.npz')

fl=open('E:/Clustering and Retreival/Week 4/4_map_index_to_word.json')
map_index_to_word=json.load(fl)
fl.close()
print tf_idf[0,:]

tf_idf=normalize(tf_idf)

def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n)

def logpdf_diagonal_gaussian(x, mean, cov):
    '''
    Compute logpdf of a multivariate Gaussian distribution with diagonal covariance at a given point x.
    A multivariate Gaussian distribution with a diagonal covariance is equivalent
    to a collection of independent Gaussian random variables.

    x should be a sparse matrix. The logpdf will be computed for each row of x.
    mean and cov should be given as 1D numpy arrays
    mean[i] : mean of i-th variable
    cov[i] : variance of i-th variable'''

    n = x.shape[0]
    dim = x.shape[1]
    assert(dim == len(mean) and dim == len(cov))

    # multiply each i-th column of x by (1/(2*sigma_i)), where sigma_i is sqrt of variance of i-th variable.
    scaled_x = x.dot( diag(1./(2*np.sqrt(cov))) )
    # multiply each i-th entry of mean by (1/(2*sigma_i))
    scaled_mean = mean/(2*np.sqrt(cov))

    # sum of pairwise squared Eulidean distances gives SUM[(x_i - mean_i)^2/(2*sigma_i^2)]
    return -np.sum(np.log(np.sqrt(2*np.pi*cov))) - pairwise_distances(scaled_x, [scaled_mean], 'euclidean').flatten()**2

def log_sum_exp(x, axis):
    '''Compute the log of a sum of exponentials'''
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log( np.sum(np.exp(x-x_max[:,np.newaxis]), axis=1) )
    else:
        return x_max + np.log( np.sum(np.exp(x-x_max), axis=0) )

def EM_for_high_dimension(data, means, covs, weights, cov_smoothing=1e-5, maxiter=int(1e3), thresh=1e-4, verbose=False):
    # cov_smoothing: specifies the default variance assigned to absent features in a cluster.
    #                If we were to assign zero variances to absent features, we would be overconfient,
    #                as we hastily conclude that those featurese would NEVER appear in the cluster.
    #                We'd like to leave a little bit of possibility for absent features to show up later.
    n = data.shape[0]
    dim = data.shape[1]
    mu = deepcopy(means)
    Sigma = deepcopy(covs)
    K = len(mu)
    weights = np.array(weights)

    ll = None
    ll_trace = []

    for i in range(maxiter):
        # E-step: compute responsibilities
        logresp = np.zeros((n,K))
        for k in xrange(K):
            logresp[:,k] = np.log(weights[k]) + logpdf_diagonal_gaussian(data, mu[k], Sigma[k])
        ll_new = np.sum(log_sum_exp(logresp, axis=1))
        if verbose:
            print(ll_new)
        logresp -= np.vstack(log_sum_exp(logresp, axis=1))
        resp = np.exp(logresp)
        counts = np.sum(resp, axis=0)

        # M-step: update weights, means, covariances
        weights = counts / np.sum(counts)
        for k in range(K):
            mu[k] = (diag(resp[:,k]).dot(data)).sum(axis=0)/counts[k]
            mu[k] = mu[k].A1

            Sigma[k] = diag(resp[:,k]).dot( data.multiply(data)-2*data.dot(diag(mu[k])) ).sum(axis=0) \
                       + (mu[k]**2)*counts[k]
            Sigma[k] = Sigma[k].A1 / counts[k] + cov_smoothing*np.ones(dim)

        # check for convergence in log-likelihood
        ll_trace.append(ll_new)
        if ll is not None and (ll_new-ll) < thresh and ll_new > -np.inf:
            ll = ll_new
            break
        else:
            ll = ll_new

    out = {'weights':weights,'means':mu,'covs':Sigma,'loglik':ll_trace,'resp':resp}

    return out
from sklearn.cluster import KMeans

np.random.seed(5)
num_clusters = 25

# Use scikit-learn's k-means to simplify workflow
kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
kmeans_model.fit(tf_idf)
centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

means = [centroid for centroid in centroids]
num_docs = tf_idf.shape[0]
weights = []
for i in xrange(num_clusters):
    # Compute the number of data points assigned to cluster i:
    num_assigned =sum(cluster_assignment==i)
    w = float(num_assigned) / num_docs
    weights.append(w)

covs = []
for i in xrange(num_clusters):
    member_rows = tf_idf[cluster_assignment==i]
    cov = (member_rows.multiply(member_rows) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
          + means[i]**2
    cov[cov < 1e-8] = 1e-8
    covs.append(cov)

out = EM_for_high_dimension(tf_idf, means, covs, weights, cov_smoothing=1e-10)
print out['loglik'] # print history of log-likelihood over time

map_index_to_word=sf.SFrame('E:/Clustering and Retreival/Week 4/4_map_index_to_word.gl')
# Fill in the blanks
def visualize_EM_clusters(tf_idf, means, covs, map_index_to_word):
    print('')
    print('==========================================================')

    num_clusters = len(means)
    for c in xrange(num_clusters):
        print('Cluster {0:d}: Largest mean parameters in cluster '.format(c))
        print('\n{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))
        
        # The k'th element of sorted_word_ids should be the index of the word 
        # that has the k'th-largest value in the cluster mean. Hint: Use np.argsort().
        sorted_word_ids =np.argsort(-means[c])

        for i in sorted_word_ids[:5]:
            print '{0: <12}{1:<10.2e}{2:10.2e}'.format(map_index_to_word['category'][i], 
                                                       means[c][i],
                                                       covs[c][i])

?np.argsort

c=0
i=0

means=out['means']
covs=out['covs']

visualize_EM_clusters(tf_idf, out['means'], out['covs'], map_index_to_word)

np.random.seed(5)
num_clusters = len(means)
num_docs, num_words = tf_idf.shape

random_means = []
random_covs = []
random_weights = []

for k in range(num_clusters):
    
    # Create a numpy array of length num_words with random normally distributed values.
    # Use the standard univariate normal distribution (mean 0, variance 1).
    # YOUR CODE HERE
    mean =np.random.normal(0,1,num_words)
    
    # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
    # YOUR CODE HERE
    cov =np.random.uniform(1,5,num_words)

    # Initially give each cluster equal weight.
    # YOUR CODE HERE
    weight =[1./num_clusters]*num_words
    
    random_means.append(mean)
    random_covs.append(cov)
    random_weights.append(weight)

out_random_init = EM_for_high_dimension(tf_idf, random_means, random_covs, weights, cov_smoothing=1e-5)
out_random_init['loglik']
out['loglik']

visualize_EM_clusters(tf_idf, out_random_init['means'], out_random_init['covs'], map_index_to_word)




