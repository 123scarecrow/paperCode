from gat import GAT 
import process

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

class BasicModel:
    def __init__(self, dim_input, dim_output, seq_length,node_num,
                 filter_num, dim_cnn_flatten, dim_fc, dim_lstm_hidden,
                 update_lr, meta_lr, meta_batch_size, update_batch_size,
                 test_num_updates, feature_size, nb_nodes,
                 hid_units, n_heads,activation=tf.nn.elu, residual=False):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.channels = dim_output
        self.node_num = node_num
        self.img_size = int(np.sqrt(self.dim_input / self.channels))

        self.dim_output = dim_output
        self.seq_length = seq_length
        self.filter_num = filter_num
        self.dim_cnn_flatten = dim_cnn_flatten
        self.dim_fc = dim_fc
        self.dim_lstm_hidden = dim_lstm_hidden

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        self.update_batch_size = update_batch_size
        self.test_num_updates = test_num_updates

        self.meta_batch_size = meta_batch_size
        
        
        self.feature_size = feature_size
        self.nb_nodes = nb_nodes          
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
      

        self.inputa = tf.placeholder(tf.float32,shape=(3,self.update_batch_size,self.seq_length,self.nb_nodes,self.feature_size))
        self.inputb = tf.placeholder(tf.float32,shape=(3,self.update_batch_size,self.seq_length,self.nb_nodes,self.feature_size))
        self.labela = tf.placeholder(tf.float32,shape=(3,self.update_batch_size,1))
        self.labelb = tf.placeholder(tf.float32,shape=(3,self.update_batch_size,1))
        #self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, ft_size))
        self.bias_ina = tf.placeholder(dtype=tf.float32,shape=(3,self.nb_nodes,self.nb_nodes))
        self.bias_inb = tf.placeholder(dtype=tf.float32,shape=(3,self.nb_nodes,self.nb_nodes))
        #self.lbl_in = tf.placeholder(dtype=tf.int32, shape=(num_graph, nb_nodes, nb_classes))
        #self.msk_in = tf.placeholder(dtype=tf.int32, shape=())
        self.attn_drop = tf.placeholder(dtype=tf.float32)
        self.ffd_drop = tf.placeholder(dtype=tf.float32)
        self.is_train = tf.placeholder(dtype=tf.bool)

    def update(self, loss, weights):
        print('【weights】',weights)
        '''
        vars = tf.trainable_variables()
        #加入gat权值更新
        for v in vars[4:]:       
            weights[v.name] = v
        print('【weights】',weights)
        '''
        #损失计算的是全部节点的预测
        grads = tf.gradients(loss, list(weights.values()))
        gradients = dict(zip(weights.keys(), grads))
        new_weights = dict(
            zip(weights.keys(), [weights[key] - self.update_lr * gradients[key] for key in weights.keys()]))
        return new_weights

    def construct_convlstm(self):
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        #k = 3
        #gat层权值
        #(node,8)
        #一维卷积 [filter_width, in_channels, out_channels]
        weights['weight'] = tf.get_variable('weight', [1, self.feature_size, self.hid_units[0]],
                                           initializer=conv_initializer, dtype=dtype)
        
        weights['att_self_weight'] = tf.get_variable('att_self_weight',[1,self.hid_units[0], 1],
                                                     initializer=conv_initializer, dtype=dtype )
        
        weights['att_neighs_weight'] = tf.get_variable('att_neighs_weight',[1,self.hid_units[0], 1],
                                                       initializer=conv_initializer, dtype=dtype)
        
        weights['bias_weight'] = tf.get_variable('bias_weight',shape=[self.hid_units[0],])
        #weights['b_conv1'] = tf.Variable(tf.zeros([self.filter_num]))
        
        #weights['weight'] = tf.get_variable('weight', [self.update_batch_size,self.feature_size ,self.hid_units[0]])
        #(8,1)
        #weights['att_self_weight'] = tf.get_variable('att_self_weight',[self.update_batch_size,self.feature_size, 1] )
        #(8,1)
        #weights['att_neighs_weight'] = tf.get_variable('att_neighs_weight',shape=[self.update_batch_size,self.nb_nodes, 1])
        #(8,)
        
        
        #gat输出层权值
        weights['output_weight'] = tf.get_variable('output_weight', [1, self.hid_units[0]*self.n_heads[0], self.feature_size],
                                           initializer=conv_initializer, dtype=dtype)
        
        weights['output_att_self_weight'] = tf.get_variable('output_att_self_weight',[1,self.feature_size, 1],
                                                            initializer=conv_initializer, dtype=dtype )
        
        weights['output_att_neighs_weight'] = tf.get_variable('output_att_neighs_weight',[1,self.feature_size, 1],
                                                              initializer=conv_initializer, dtype=dtype)
        
        weights['output_bias_weight'] = tf.get_variable('output_bias_weight',shape=[self.feature_size,])
        '''
        weights['output_weight'] = tf.get_variable('output_weight', [self.update_batch_size,self.hid_units[0]*self.n_heads[0] , self.feature_size])
        #(8,1)
        weights['output_att_self_weight'] = tf.get_variable('output_att_self_weight',[self.update_batch_size,self.nb_nodes, 1] )
        #(8,1)
        weights['output_att_neighs_weight'] = tf.get_variable('output_att_neighs_weight',[self.update_batch_size,self.nb_nodes, 1])
        #(8,)
        
        '''
        #LSTM
        #（640,512）
        weights['kernel_lstm'] = tf.get_variable('kernel_lstm', [self.feature_size + self.dim_lstm_hidden, 4 * self.dim_lstm_hidden])
        #（512,）
        weights['b_lstm'] = tf.Variable(tf.zeros([4 * self.dim_lstm_hidden]))
        #()
        weights['b_fc2'] = tf.Variable(tf.zeros([self.dim_output]))

        return weights

    def lstm(self, inp, weights,bias_in,feature_size, nb_nodes,                          
                                hid_units, n_heads,
                                residual, activation):
        print("Initializing LSTM...")
        def lstm_block(linp, pre_state, kweight, bweight, activation):
            
            sigmoid = math_ops.sigmoid
            one = constant_op.constant(1, dtype=dtypes.int32)
            c, h = pre_state
            print('【c】',c.shape)#(128,128)\\ （x，x）
            print('【h】',h.shape)#(128,128)\\(x,x)
            print('【linp】',linp)#(1,512)\\(n,f)
            print('【kweight】',kweight)#()
            #（42,18+x）*（18+x，h）
            #（sn，sm+h1）*（640,512）\\（n,f+x）*(f+x,h)
            gate_inputs = math_ops.matmul(
                array_ops.concat([linp, h], 1), kweight)# 按照第二维度相接
            #（1,h）+（h,）=(1,h)
            print('【gate_inputs】',gate_inputs)
            gate_inputs = nn_ops.bias_add(gate_inputs, bweight)
            print('【gate_inputs】',gate_inputs)
            i, j, f, o = array_ops.split(
                value=gate_inputs, num_or_size_splits=4, axis=one)

            forget_bias_tensor = constant_op.constant(1.0, dtype=f.dtype)

            add = math_ops.add
            multiply = math_ops.multiply
            new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),
                        multiply(sigmoid(i), activation(j)))
            new_h = multiply(activation(new_c), sigmoid(o))

            new_state = [new_c, new_h]
            print('【new_h】',new_h)
            return new_h, new_state

        # unstack对矩阵分解
        # transpose多维矩阵转置 perm=[1,0,2] 例如：2*3*4 -> 3*2*4
        #转换成3个（2*4）的list ，3表示时间
        #time*（samplenum,features）
        inp = tf.unstack(tf.transpose(inp, perm=[1, 0, 2,3]))
        print('【inp】',inp)
        #(128,128)
        state = [tf.zeros([self.update_batch_size, self.dim_lstm_hidden]),
                 tf.zeros([self.update_batch_size, self.dim_lstm_hidden])]
        print('【state】',state)
        output = None
        
        for t in range(self.seq_length):
            mean,var = tf.nn.moments(inp[t], axes = [0])        
            epsilon = 0.001
            W = tf.nn.batch_normalization(inp[t], mean, var, 0.0, 1.0, epsilon)
            gat_outputs = GAT.inference(W,weights,bias_in ,feature_size, nb_nodes, self.is_train, self.attn_drop,
                                    self.ffd_drop ,bias_in,                            
                                    hid_units, n_heads,
                                    activation, residual)
            #(batchsize,nodenum,features)
            lstm_inputs = gat_outputs[:,self.node_num,:]
         
            output, state = lstm_block(lstm_inputs, state,
                                       weights['kernel_lstm'], weights['b_lstm'],
                                       tf.nn.tanh)
  
        return output

    def forward_convlstm(self, inp, weights,bias_in,feature_size, nb_nodes,                          
                                hid_units, n_heads,
                                residual, activation):
        print("Initializing forward_convlstm...")
        #inp = tf.reshape(inp, [16,32,16])
        print(inp.shape)
        
        #gat_outputs(nodeNum,features)
        #print('【gat_outputs】',gat_outputs)
        #选取一个站点预测
        #gat_outputs = gat_outputs[:,node_num,:]
        #(samplenum，seq_len，ft_sz)
        #gat_outputs = tf.reshape(gat_outputs, [-1, self.seq_length, self.dim_fc])#(?,8,512)#（32，2，8）
        
        lstm_outputs = self.lstm(inp, weights,bias_in,feature_size, nb_nodes,                          
                                hid_units, n_heads,
                                residual, activation)
        print('【lstm_outputs】',lstm_outputs)
        return lstm_outputs


class STDN(BasicModel):
    def __init__(self, dim_input, dim_output, seq_length,node_num,
                 filter_num, dim_cnn_flatten, dim_fc, dim_lstm_hidden,
                 update_lr, meta_lr, meta_batch_size, update_batch_size,
                 test_num_updates, feature_size, nb_nodes, 
                 hid_units, n_heads,activation=tf.nn.elu, residual=False):
        print("Initializing STDN...")
        BasicModel.__init__(self, dim_input, dim_output, seq_length,node_num,
                 filter_num, dim_cnn_flatten, dim_fc, dim_lstm_hidden,
                 update_lr, meta_lr, meta_batch_size, update_batch_size,
                 test_num_updates, feature_size, nb_nodes, 
                 hid_units, n_heads,activation=tf.nn.elu, residual=False)

    def loss_func(self, pred, label):
        print('【pred】',pred)
        print('【label】',label)
        pred = tf.reshape(pred, [-1])
        label = tf.reshape(label, [-1])
        return tf.reduce_mean(tf.square(pred - label))

    def construct_model(self):
        with tf.variable_scope('model', reuse=None):
            with tf.variable_scope('maml', reuse=None):
                self.weights = weights = self.construct_convlstm()
                weights['fc2'] = tf.Variable(tf.random_normal(
                    [self.dim_lstm_hidden, self.dim_output]), name='fc6')   # output layer

            num_updates = self.test_num_updates
            print("Initializing construct_model...")
            def task_metalearn(inp):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb ,bias_ina,bias_inb = inp
                epsilon = 0.001
                mean,var = tf.nn.moments(labela, axes = [0])                       
                labela = tf.nn.batch_normalization(labela, mean, var, 0.0, 1.0, epsilon)
                mean,var = tf.nn.moments(labelb, axes = [0])                      
                labelb = tf.nn.batch_normalization(labelb, mean, var, 0.0, 1.0, epsilon)
                task_outputbs, task_lossesb = [], []
                #inputa是support集
                task_outputa = self.forward(inputa,bias_ina, weights)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                fast_weights = self.update(task_lossa, weights)
                #inputb是query集
                output = self.forward(inputb,bias_inb,fast_weights)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))
                #对于每一个task的迭代次数
                for j in range(num_updates - 1):
                    loss = self.loss_func(self.forward(inputa,bias_ina, fast_weights), labela)
                    fast_weights = self.update(loss, fast_weights)

                    output = self.forward(inputb,bias_inb, fast_weights)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))
                #每个task的损失
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                return task_output

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            print(self.inputa.shape)
            inputs = (self.inputa, self.inputb, self.labela, self.labelb,self.bias_ina,self.bias_inb)
            result = tf.map_fn(task_metalearn,
                               elems=inputs,
                               dtype=out_dtype,
                               parallel_iterations=self.meta_batch_size)
            outputas, outputbs, lossesa, lossesb = result

        # Performance & Optimization
        self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(self.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.meta_batch_size)
                                              for j in range(num_updates)]
        self.total_rmse1 = tf.sqrt(lossesa)
        self.total_rmse2 = [tf.sqrt(total_losses2[j]) for j in range(num_updates)]

        self.outputas, self.outputbs = outputas, outputbs
        self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
        self.metatrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_losses2[num_updates-1])

        maml_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "model/maml")
        self.finetune_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1, var_list=maml_vars)

    def forward(self, inp,bias_in, weights):
        print("Initializing forward...")
        print(inp.shape)
        convlstm_outputs = self.forward_convlstm(inp, weights,bias_in,self.feature_size, self.nb_nodes,                          
                                self.hid_units,self.n_heads,
                                self.residual, self.activation)
        
        preds = tf.nn.tanh(tf.matmul(convlstm_outputs, weights['fc2']) + weights['b_fc2'])
        print('【preds】',preds)
        return preds