import numpy as np
import scipy
import scipy.sparse as sp
import torch
import tensorflow as tf
import random
import data.io as io
import copy
import arguments
args = arguments.parse_args()


def known_unknown_split(
        idx: np.ndarray, nknown: int = 1500, seed: int = 4143496719):
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx

def exclude_idx(idx: np.ndarray, idx_exclude_list):
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])

def train_stopping_split(
        idx: np.ndarray, labels: np.ndarray, ntrain_per_class: int = 20,
        nstopping: int = 500, seed: int = 2413340114):
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    for i in range(max(labels) + 1):
        train_idx_split.append(rnd_state.choice(
                idx[labels == i], ntrain_per_class, replace=False))
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
            exclude_idx(idx, [train_idx]),
            nstopping, replace=False)
    return train_idx, stopping_idx

def gen_splits(labels: np.ndarray, idx_split_args,
        test: bool = False):
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(
            all_idx, idx_split_args['nknown'])
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args['nknown']
    train_idx, stopping_idx = train_stopping_split(
            known_idx, labels[known_idx], **stopping_split_args)
    if test:
        val_idx = unknown_idx
    else:
        val_idx = exclude_idx(known_idx, [train_idx, stopping_idx])
    return train_idx, stopping_idx, val_idx

def load_data(graph_name = 'cora_ml', lbl_noise=0, str_noise_rate=0.1, seed = 2144199730):
    dataset = io.load_dataset(graph_name)
    dataset.standardize(select_lcc=True)
    features = dataset.attr_matrix
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    #features = sparse_mx_to_torch_sparse_tensor(features)
    labels = dataset.labels
    adj = dataset.adj_matrix
    adj = str_noise(adj, labels, str_noise_rate, seed)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if graph_name == 'ms_academic':
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 5000, 'seed': seed}
    else:
        idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': seed}
    idx_train, idx_val, idx_test = gen_splits(labels, idx_split_args, test=True)

    n_class = max(labels) + 1
    a, b = getMaxAB(labels)
    #labels = add_label_noise(idx_train, labels, lbl_noise, seed)
    #labels = gradient_max(adj, idx_train, labels, lbl_noise)#adj, idx_train, labels, noise_num
    #labels[idx_train] = noisify(labels[idx_train], n_class, noise_rate=noise_rate, random_state=10, noise_type=noise_type)
    featuresLaf = dataset.attr_matrix
    adjLaf = dataset.adj_matrix
    adjLaf = str_noise(adjLaf, labels, str_noise_rate, seed)
    adjLaf = adjLaf + sp.eye(adjLaf.shape[0])
    labels = LafAK(adjLaf, featuresLaf, idx_train, idx_test, idx_val, labels, lbl_noise, a, b)

    # unlabeled, nodesab_train, nodesa_un, nodesb_un, nodesab_un, nodesab_all, \
    # split_train, split_test, split_val, split_unlabeled = split_ab(adj, labels, idx_train, idx_test, idx_val, a, b)

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)


    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_features(mx): #row_normalize(mx)
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def split_ab(adj, labels, idx_train, idx_test, idx_val, a, b):
    unlabeled = [i for i in range(adj.shape[0]) if i not in idx_train]
    untest_id = [unlabeled.index(x) for x in idx_test]  # test在unlabel里的下标
    unlabeled_ori = list(unlabeled)
    if labels.max() + 1 == 2:
        nodesab_all = range(labels.shape[0])
        nodesab_train = np.array(idx_train)
        return

    labela = np.squeeze(np.argwhere(labels == a))
    labelb = np.squeeze(np.argwhere(labels == b))
    labela_train = get_intersection(labela, idx_train)
    labelb_train = get_intersection(labelb, idx_train)
    nodesab_train = np.hstack((labela_train, labelb_train))

    labela_test = get_intersection(labela, idx_test)
    labelb_test = get_intersection(labelb, idx_test)
    nodesab_test = np.hstack((labela_test, labelb_test))
    labela_val = get_intersection(labela, idx_val)
    labelb_val = get_intersection(labelb, idx_val)
    nodesab_val = np.hstack((labela_val, labelb_val))

    nodesa_un = get_intersection(labela, unlabeled)
    nodesb_un = get_intersection(labelb, unlabeled)
    nodesab_un = np.hstack((nodesa_un, nodesb_un))
    nodesab_all = np.hstack((nodesab_train, nodesab_un))
    newIdx = nodesab_all.tolist()
    split_train = [newIdx.index(x) for x in nodesab_train]
    split_unlabeled = [newIdx.index(x) for x in nodesab_un]
    split_test = [newIdx.index(x) for x in nodesab_test]
    split_val = [newIdx.index(x) for x in nodesab_val]
    return unlabeled, nodesab_train, nodesa_un, nodesb_un, nodesab_un, nodesab_all, split_train, split_test, \
           split_val, split_unlabeled

def closedForm_bin(adj, features, labels, idx_train, idx_test, idx_val, a, b):
    # closed form of GCN
    #a, b = getMaxAB(labels)
    K = getK_GCN(adj, features, labels, idx_train, idx_test, idx_val, a, b)
    y_pred = np.dot(K, labels[idx_train])  # dot(K,y_L)
    return y_pred

def getMaxAB(labels):
    count = []
    #class_index = labels.max()
    for i in range(labels.max()+1):
        count.append(labels[labels == i].shape[0])
    # print(count)
    #count_sorted = sorted(count)
    #x = count.index(count_sorted[6])

    a = count.index(max(count))
    count[count.index(max(count))] = 0
    b = count.index(max(count))
    # count[count.index(max(count))] = 0
    # c = count.index(max(count))
    # count[count.index(max(count))] = 0
    # d = count.index(max(count))

    return a, b#, c, d

def getK_GCN(adj, features, labels, idx_train, idx_test, idx_val, a, b):
    #A = preprocess_graph(adj).tocsr()#1.add self-loop 2.sys-normalize
    #a, b = getMaxAB(labels)
    unlabeled, nodesab_train, nodesa_un, nodesb_un, nodesab_un, nodesab_all, split_train, split_test, \
           split_val, split_unlabeled = split_ab(adj, labels, idx_train, idx_test, idx_val, a, b)
    A = normalize_adj(adj[nodesab_all,:][:,nodesab_all]).tocsr()
    X = normalize_features(features[nodesab_all]).tocsr()
    X_bar = A.dot(A).dot(X).tocsr()
    X_bar_l = X_bar[split_train,:]
    tmp = X_bar_l.T.dot(X_bar_l)
    gcnL2 = 0.005
    tmp = sp.csr_matrix(scipy.linalg.pinv(tmp.toarray()+ gcnL2*np.identity(tmp.shape[0])))
    tmp = tmp.dot(X_bar_l.T)
    K = X_bar.dot(tmp)[split_unlabeled,:].toarray()
    return K

def get_intersection(a,b):
    return list(set(a).intersection(set(b)))

def LafAK(adj, features, idx_train, idx_test, idx_val, labels, noise_num, a, b):
    #定义a,b的值和序列，求alpha, 根据alpha求d_y, d_y确定位置，getMaxAB(labels)中a转为b
    #a, b = getMaxAB(labels)
    unlabeled, nodesab_train, nodesa_un, nodesb_un, nodesab_un, nodesab_all, split_train, split_test, \
           split_val, split_unlabeled = split_ab(adj, labels, idx_train, idx_test, idx_val, a, b)
    #a, b = getMaxAB(labels)
    tau = 4
    K = getK_GCN(adj, features, labels, idx_train, idx_test, idx_val, a, b)
    #y_pred = closedForm_bin(adj, features, labels, idx_train, idx_test, idx_val, a, b)

    y_l = np.reshape(labels[split_train], (len(split_train), 1))
    y_u = np.reshape(labels[split_unlabeled], (len(split_unlabeled), 1))
    alpha = tf.Variable(name='alpha', initial_value=(0.5 * np.ones_like(y_l)),dtype='float32')
    epsilon = tf.placeholder(tf.float32, name='epsilon')
    tmp = tf.exp((tf.log(alpha / (tf.constant(1.0, dtype='float32') - alpha)) + epsilon) / tf.constant(0.5))
    z = tf.constant(2.0, dtype='float32') / (tf.constant(1.0, dtype='float32') + tmp) - tf.constant(1.0,
                                                                                                    dtype='float32')  # normalize z from [0, 1] to [-1, 1]
    y_l_tmp = tf.constant(y_l, dtype='float32') * z
    # approximated closed form of GCN
    y_u_preds = tf.nn.tanh(tau * tf.matmul(tf.constant(K, dtype='float32'), y_l_tmp))
    loss = tf.reduce_mean(y_u_preds * tf.constant(y_u, dtype='float32'))
    optimizer = tf.train.GradientDescentOptimizer(1.0e-4)
    opt_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        for step in range(args.epochs):
            ep = np.random.gumbel(len(y_l)) - np.random.gumbel(len(y_l))  # reparameterization trick
            sess.run(opt_op, feed_dict={epsilon: ep})  # update alpha
        alpha = sess.run(alpha, feed_dict={epsilon: ep})  # return alpha
    alpha = np.reshape(np.array(alpha), (len(split_train)))
    idx = np.argsort(alpha)[::-1]
    d_y = np.ones_like(labels[split_train])
    count = 0
    flip = {}
    f = flip.keys()
    # flip_key = list(flip.keys())
    for i in idx:
        if alpha[i] > 0.5 and count < noise_num :
            d_y[i] = -1
            count += 1
            flip[split_train[i]] = alpha[i]
            if count == noise_num:
                break

    flipNodes = np.array(nodesab_train)[d_y != 1]
    for i in flipNodes:
        if i not in nodesab_train.tolist():
            print("wrong flipNodes!")
            exit()
        if labels[i] == a:
            labels[i] = b
        elif labels[i] == b:
            labels[i] = a
        else:
            print("wrong flipNodes!")
            exit()
    #l = labels
    return labels  #alpha: 翻转最大alpha对应的元素

def str_noise(adj, labels, noise_rate, seed = 0):
    if noise_rate > 1.0:
        return adj
    idx = np.arange(len(labels))
    adj = adj.tocoo().astype(np.float32)

    row = adj.row
    col = adj.col

    upper_edge = np.arange(len(row))[row < col]
    idx_upper_edge = np.arange(len(upper_edge))
    good_edge_idx = idx_upper_edge[labels[row[upper_edge]] == labels[col[upper_edge]]]
    bad_edge_idx = idx_upper_edge[labels[row[upper_edge]] != labels[col[upper_edge]]]

    origin_noise_rate = len(bad_edge_idx) / len(idx_upper_edge)

    rnd_state = np.random.RandomState(seed)
    random.seed(seed)
    if origin_noise_rate > noise_rate:
        sub_num = int(len(upper_edge) * (origin_noise_rate - noise_rate))
        inv_idx = rnd_state.choice(bad_edge_idx, sub_num, replace=False)
        for i in inv_idx:
            row_i = row[[upper_edge[i]]]
            col_i = col[[upper_edge[i]]]
            if random.random() > 0.5:
                lbl = labels[row_i]
                col_new = rnd_state.choice(idx[labels == lbl], 2, replace=False)
                if col_new[0] != row_i:
                    col_new = col_new[0]
                else:
                    col_new = col_new[1]
                col[[upper_edge[i]]] = col_new
                for j in range(len(row)):
                    if row[j] == col_i and col[j] == row_i:
                        row[j] = col_new
                        break
            else:
                lbl = labels[col_i]
                row_new = rnd_state.choice(idx[labels == lbl], 2, replace=False)
                if row_new[0] != col_i:
                    row_new = row_new[0]
                else:
                    row_new = row_new[1]
                row[[upper_edge[i]]] = row_new
                for j in range(len(row)):
                    if row[j] == col_i and col[j] == row_i:
                        col[j] = row_new
                        break
    else:
        add_num = int(len(upper_edge) * (noise_rate - origin_noise_rate))
        inv_idx = rnd_state.choice(good_edge_idx, add_num, replace=False)
        for i in inv_idx:
            row_i = row[[upper_edge[i]]]
            col_i = col[[upper_edge[i]]]
            if random.random() > 0.5:
                lbl = labels[row_i]
                col_new = rnd_state.choice(idx[labels != lbl], 1, replace=False)
                col_new = col_new[0]
                col[[upper_edge[i]]] = col_new
                for j in range(len(row)):
                    if row[j] == col_i and col[j] == row_i:
                        row[j] = col_new
                        break
            else:
                lbl = labels[col_i]
                row_new = rnd_state.choice(idx[labels != lbl], 1, replace=False)
                row_new = row_new[0]
                row[[upper_edge[i]]] = row_new
                for j in range(len(row)):
                    if row[j] == col_i and col[j] == row_i:
                        col[j] = row_new
                        break

    adj.row = row
    adj.col = col
    return adj

def get_noise_rate(adj, labels):
    indices = adj._indices().numpy()
    upper = indices[0, :] > indices[1, :]
    upper_indices = indices[:, upper]

    bad_num = 0
    for (i, j) in np.transpose(upper_indices):
        if labels[i].item() != labels[j].item():
            bad_num += 1

    return bad_num / upper_indices.shape[1]

