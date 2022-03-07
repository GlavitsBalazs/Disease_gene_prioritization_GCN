from __future__ import division
from __future__ import print_function
from operator import itemgetter
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
import matplotlib.pyplot as plt
import h5py
import pickle

from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel
from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.utility import rank_metrics, preprocessing

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

np.random.seed(0)

def tsne_visualization(matrix):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    plt.figure(dpi=300)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0,
            n_iter=1000)
    tsne_results = tsne.fit_transform(matrix)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def draw_graph(adj_matrix):
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    pos = nx.spring_layout(G, iterations=100)
    d = dict(nx.degree(G))
    nx.draw(G, pos, node_color=range(3215), nodelist=d.keys(), 
        node_size=[v*20+20 for v in d.values()], cmap=plt.cm.Dark2)
    plt.show()

def bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0):
    """BEDROC metric implemented according to Truchon and Bayley.
    Copyright (C) 2015-2016 Rich Lewis <rl403@cam.ac.uk>
    License: 3-clause BSD

    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.

    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.

    Args:
        y_true (array_like):
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (array_like):
            Prediction values.
        decreasing (bool):
            True if high values of ``y_pred`` correlates to positive class.
        alpha (float):
            Early recognition parameter.

    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
     """

    assert len(y_true) == len(y_pred), \
        'The number of scores must be equal to the number of labels'

    N = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    m_rank = (y_true[order] == 1).nonzero()[0]
    s = np.sum(np.exp(-alpha * m_rank / N))
    r_a = n / N
    rand_sum = r_a * (1 - np.exp(-alpha)) / (np.exp(alpha / N) - 1)
    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * r_a))
    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))
    return s * fac / rand_sum + cte

def get_accuracy_scores(edges_pos, edges_neg, edge_type, name=None):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)

        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=200)
    bedroc_sc = bedroc_score(labels_all, preds_all)
    if name!=None:
        with open(name, 'wb') as f:
            pickle.dump([labels_all, preds_all], f)
    return roc_sc, aupr_sc, apk_sc, bedroc_sc


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

def network_edge_threshold(network_adj, threshold):
    edge_tmp, edge_value, shape_tmp = preprocessing.sparse_to_tuple(network_adj)
    preserved_edge_index = np.where(edge_value>threshold)[0]
    preserved_network = sp.csr_matrix(
        (edge_value[preserved_edge_index], 
        (edge_tmp[preserved_edge_index,0], edge_tmp[preserved_edge_index, 1])),
        shape=shape_tmp)
    return preserved_network


def get_prediction(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    return 1. / (1 + np.exp(-rec))

gene_phenes_path = './data_prioritization/genes_phenes.mat'
f = h5py.File(gene_phenes_path, 'r')
gene_network_adj = sp.csc_matrix((np.array(f['GeneGene_Hs']['data']),
    np.array(f['GeneGene_Hs']['ir']), np.array(f['GeneGene_Hs']['jc'])),
    shape=(12331,12331))
gene_network_adj = gene_network_adj.tocsr()
disease_network_adj = sp.csc_matrix((np.array(f['PhenotypeSimilarities']['data']),
    np.array(f['PhenotypeSimilarities']['ir']), np.array(f['PhenotypeSimilarities']['jc'])),
    shape=(3215, 3215))
disease_network_adj = disease_network_adj.tocsr()

disease_network_adj = network_edge_threshold(disease_network_adj, 0.2)


dg_ref = f['GenePhene'][0][0]
gene_disease_adj = sp.csc_matrix((np.array(f[dg_ref]['data']),
    np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
    shape=(12331, 3215))
gene_disease_adj = gene_disease_adj.tocsr()

novel_associations_adj = sp.csc_matrix((np.array(f['NovelAssociations']['data']),
    np.array(f['NovelAssociations']['ir']), np.array(f['NovelAssociations']['jc'])),
    shape=(12331,3215))

gene_feature_path = './data_prioritization/GeneFeatures.mat'
f_gene_feature = h5py.File(gene_feature_path,'r')
gene_feature_exp = np.array(f_gene_feature['GeneFeatures'])
gene_feature_exp = np.transpose(gene_feature_exp)
gene_network_exp = sp.csc_matrix(gene_feature_exp)

row_list = [3215, 1137, 744, 2503, 1143, 324, 1188, 4662, 1243]
gene_feature_list_other_spe = list()
for i in range(1,9):
    dg_ref = f['GenePhene'][i][0]
    disease_gene_adj_tmp = sp.csc_matrix((np.array(f[dg_ref]['data']),
        np.array(f[dg_ref]['ir']), np.array(f[dg_ref]['jc'])),
        shape=(12331, row_list[i]))
    gene_feature_list_other_spe.append(disease_gene_adj_tmp)

disease_tfidf_path = './data_prioritization/clinicalfeatures_tfidf.mat'
f_disease_tfidf = h5py.File(disease_tfidf_path)
disease_tfidf = np.array(f_disease_tfidf['F'])
disease_tfidf = np.transpose(disease_tfidf)
disease_tfidf = sp.csc_matrix(disease_tfidf)

dis_dis_adj_list= list()
dis_dis_adj_list.append(disease_network_adj)

val_test_size = 0.1
n_genes = 12331
n_dis = 3215
n_dis_rel_types = len(dis_dis_adj_list)
gene_adj = gene_network_adj
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

gene_dis_adj = gene_disease_adj
dis_gene_adj = gene_dis_adj.transpose(copy=True)

dis_degrees_list = [np.array(dis_adj.sum(axis=0)).squeeze() for dis_adj in dis_dis_adj_list]

adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_dis_adj],
    (1, 0): [dis_gene_adj],
    (1, 1): dis_dis_adj_list + [x.transpose(copy=True) for x in dis_dis_adj_list],
}
degrees = {
    0: [gene_degrees, gene_degrees],
    1: dis_degrees_list + dis_degrees_list,
}

gene_feat = sp.hstack(gene_feature_list_other_spe+[gene_feature_exp])
gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

dis_feat = disease_tfidf
dis_nonzero_feat, dis_num_feat = dis_feat.shape
dis_feat = preprocessing.sparse_to_tuple(dis_feat.tocoo())

num_feat = {
    0: gene_num_feat,
    1: dis_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: dis_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: dis_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
# edge_type2decoder = {
#     (0, 0): 'bilinear',
#     (0, 1): 'bilinear',
#     (1, 0): 'bilinear',
#     (1, 1): 'bilinear',
# }

edge_type2decoder = {
    (0, 0): 'innerproduct',
    (0, 1): 'innerproduct',
    (1, 0): 'innerproduct',
    (1, 1): 'innerproduct',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

if __name__ == '__main__':

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', 0.001, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
    flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
    flags.DEFINE_boolean('bias', True, 'Bias term.')

    print("Defining placeholders")
    placeholders = construct_placeholders(edge_types)

    print("Create minibatch iterator")
    minibatch = EdgeMinibatchIterator(
        adj_mats=adj_mats_orig,
        feat=feat,
        edge_types=edge_types,
        batch_size=FLAGS.batch_size,
        val_test_size=val_test_size
    )

    print("Create model")
    model = DecagonModel(
        placeholders=placeholders,
        num_feat=num_feat,
        nonzero_feat=nonzero_feat,
        edge_types=edge_types,
        decoders=edge_type2decoder,
    )

    print("Create optimizer")
    with tf.name_scope('optimizer'):
        opt = DecagonOptimizer(
            embeddings=model.embeddings,
            latent_inters=model.latent_inters,
            latent_varies=model.latent_varies,
            degrees=degrees,
            edge_types=edge_types,
            edge_type2dim=edge_type2dim,
            placeholders=placeholders,
            batch_size=FLAGS.batch_size,
            margin=FLAGS.max_margin
        )

    print("Initialize session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {}
    saver = tf.train.Saver()
    saver.restore(sess,'./model/model.ckpt')
    feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
    feed_dict = minibatch.update_feed_dict(
        feed_dict=feed_dict,
        dropout=FLAGS.dropout,
        placeholders=placeholders)

    roc_score, auprc_score, apk_score, bedroc = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[3])
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[3])
    print("Edge type:", "%04d" % 3, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % 3, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % 3, "Test AP@k score", "{:.5f}".format(apk_score))
    print("Edge type:", "%04d" % 3, "Test BEDROC score", "{:.5f}".format(bedroc))
    print()

    prediction = get_prediction(minibatch.test_edges, minibatch.test_edges_false, 
    	minibatch.idx2edge_type[3])

    print('Saving result...')
    np.save('./result/prediction.npy', prediction)
