import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import sys
import pickle as pkl
import networkx as nx

conv1d = tf.layers.conv1d

#注意力层
def attn_head(seq,weights,bias_mat,out_sz ,activation, in_drop=0.0, coef_drop=0.0, residual=False,is_output=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            #防止过拟合，保留seq中1.0 - in_drop个数，保留的数并变为1/1.0 - in_drop
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        #将原始节点特征 seq 进行变换得到了 seq_fts。这里，作者使用卷积核大小为 1 的 1D 卷积模拟投影变换，
        # 投影变换后的维度为 out_sz。注意，这里投影矩阵 W是所有节点共享，所以 1D 卷积中的多个卷积核也是共享的。
        #seq_fts 的大小为 [num_graph, num_node, out_sz]
        #seq = tf.reshape(seq,(1,seq.shape[0],seq.shape[1]))
        if is_output == True:
            conv1_weight = weights['output_weight']
            att_self_weight = weights['output_att_self_weight']
            att_neighs_weight = weights['output_att_neighs_weight']
            bias_weight = weights['output_bias_weight']
        else:
            conv1_weight = weights['weight']
            att_self_weight = weights['att_self_weight']
            att_neighs_weight = weights['att_neighs_weight']
            bias_weight = weights['bias_weight']
        
        print('【seq】',seq.shape)
        #(1, 2708, 1433)
        print('【weight】',conv1_weight)
        #seq_fts = tf.layers.conv1d(seq, out_sz,1, use_bias=False) #filters=feature_size,kernel_size = 1,
        seq_fts = tf.nn.conv1d(seq, conv1_weight, stride=1, padding='VALID')
        #seq_fts = tf.matmul(seq, conv1_weight )
        print('【seq_fts】',seq_fts)
        #(1, 2708, 8)
        # simplest self-attention possible
        # f_1 和 f_2 维度均为 [num_graph, num_node, 1]
        #(1, 2708, 1)
        f_1 = tf.nn.conv1d(seq_fts, att_self_weight, stride=1, padding='VALID')
        f_2 = tf.nn.conv1d(seq_fts, att_neighs_weight, stride=1, padding='VALID')
        #f_1 = tf.reduce_sum(seq_fts * att_self_weight, axis=-1, keep_dims=True)  # None head_num 1
        
        #f_2 = tf.reduce_sum(seq_fts * att_neighs_weight, axis=-1, keep_dims=True)
        
        #f_1 = tf.layers.conv1d(seq_fts, 1, 1)  #节点投影
        #f_2 = tf.layers.conv1d(seq_fts, 1, 1)  #邻居投影
        

        #将 f_2 转置之后与 f_1 叠加，通过广播得到的大小为 [num_graph, num_node, num_node] 的 logits
        logits = f_1 + tf.transpose(f_2, [0,2,1])#注意力矩阵
        #(1, 2708, 2708)
        print('【logits】',logits)
        #+biase_mat是为了对非邻居节点mask,归一化的注意力矩阵
        #邻接矩阵的作用，把和中心节点没有链接的注意力因子mask掉
        mask = -10e9 * (1.0 - bias_mat)
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + mask)
        
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
         #将 mask 之后的注意力矩阵 coefs 与变换后的特征矩阵 seq_fts 相乘，
         # 即可得到更新后的节点表示 vals。
        #(1, 2708, 2708)*(1,2708,8)=(1, 2708, 8)
        print('【coefs】',coefs)
        vals = tf.matmul(coefs, seq_fts)
        #ret = tf.contrib.layers.bias_add(vals)
        #ret = vals + weights['bias_weight']
        ret = tf.nn.bias_add(vals, bias_weight)
        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq
        print(activation)
        return activation(ret)  # activation

class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # training op
        train_op = opt.minimize(loss+lossL2)
        
        return train_op

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)

##########################
# Adapted from tkipf/gcn #
##########################

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss=tf.reduce_mean(loss,axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)
        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure
class GAT(BaseGAttN):
    def inference(inputs,weights,bias_in, feature_size, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        #将多头输出连接在一起concat，有n个输入层
        for _ in range(n_heads[0]):
            attns.append(attn_head(inputs,weights, bias_mat=bias_in,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False,is_output=False))
        #(1, 2708, 64)
        h_1 = tf.concat(attns, axis=-1)
        
        print('【hid_units】',len(hid_units))
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(attn_head(h_1,weights, bias_mat=bias_in,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual,is_output=False))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        print('【h_1】',h_1)
        for i in range(n_heads[-1]):
            out.append(attn_head(h_1,weights, bias_mat=bias_in,
                out_sz=feature_size, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False,is_output=True))
        
        logits = tf.add_n(out) / n_heads[-1]
        print('【logits】',logits)
        return logits