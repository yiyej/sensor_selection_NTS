#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:12:35 2019

@author: yiye
"""

# The codes are the modified version of the original software:
# https://github.com/mdeff/cnn_graph, with the permission granted, see the LICENSE file.

from . import graph

import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
import scipy.sparse
import numpy as np
import os, time, collections, shutil


class base_model(object):
    
    def __init__(self):
        self.regularizers = [0.0]
    
    # High-level interface which runs the constructed computational graph.
    
    def predict(self, data, targets, covariates = None, sess=None):
        loss = 0
        size = (data.shape[0], targets.shape[1])
        predictions = np.empty(size)
        sess = self._get_session(sess)
        for begin in range(0, data.shape[0], self.batch_size):
            end = begin + self.batch_size
            end = min([end, data.shape[0]])
            
            batch_data = np.zeros((self.batch_size, *data.shape[1:]))
            batch_targets = np.zeros((self.batch_size, targets.shape[1]))
            tmp_targets = targets[begin:end,:]
            tmp_data = data[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices 
            if type(tmp_targets) is not np.ndarray:
                tmp_targets = tmp_targets.toarray()              
            
            if data.shape[0] < begin + self.batch_size:
                ind = list(range(end-begin))*int(self.batch_size/(end-begin)+1)
                ind = ind[:self.batch_size]
                batch_data = tmp_data[ind,]
                batch_targets = tmp_targets[ind,]
            else:
                batch_data = tmp_data  
                batch_targets = tmp_targets
                                            
            feed_dict = {self.ph_data: batch_data, self.ph_targets: batch_targets, self.ph_rate_fc: 0}
            
            if self.D is not None:
                batch_covariates = np.zeros((self.batch_size, covariates.shape[1]))
                tmp_covariates = covariates[begin:end,:]
                if type(tmp_covariates) is not np.ndarray:
                    tmp_covariates = tmp_covariates.toarray()  
                    
                if data.shape[0] < begin + self.batch_size:
                    batch_covariates = tmp_covariates[ind,]
                else:
                    batch_covariates = tmp_covariates                          
                feed_dict[self.ph_covariates] = batch_covariates
                
            if self.selecting_mode:
                feed_dict[self.ph_rate_input] = 0

            batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
            loss += batch_loss
            predictions[begin:end,:] = batch_pred[:end-begin,:]
            
        return predictions, loss
        
    def evaluate(self, data, targets, covariates = None, sess=None):
        """ Test time.

        sess: the session in which the model has been trained.
        data: size N x M x (H+1)
            N: number of signals (samples)
            M: number of vertices (features)
            H: time lag.
        covariates: size N x D.
            D: number of external features
        targets: size N x C
            C: number of targets
        """
        t_process, t_wall = time.process_time(), time.time()
        sess = self._get_session(sess)
        predictions, loss = self.predict(data, targets, covariates, sess)
        r2 = r2_score(targets, predictions)
        
        string = 'r2: {:.2f}, loss: {:.2e}'.format(r2, loss)
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string, predictions, r2, loss

    def fit(self, train_data, train_targets, val_data = None, val_targets = None, continue_training = False, train_cova = None, val_cova = None):
        """ Training time.

        train_data: size N1 x M x (H+1)
                N1: number of signals (samples) in training set
                M: number of vertices (features)
                H: time lag.
        train_targets: size N1 x C
        val_data: size N2 x M x (H+1)
                N2: number of signals (samples) in validation set
        val_target: size N2 x C
        train_cova: size N1 x D.
                D: number of external features
        val_cova: size N2 x D
        continue_training: if True, restore the parameter values saved in checkpoint folder, continue training,
                           else if False, intialize the parameter values and start training.
        
        When fitting the selection net with masking L1 regularization method, 
        at each epoch, the mask weight distribution will be plotted and saved in the working directory.
        """
        t_process, t_wall = time.process_time(), time.time()
        if continue_training:
            sess = self._get_session(sess=None)
        else:
            sess = tf.Session(graph=self.graph)
            K.set_session(sess)
        #shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        #writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        if not continue_training: 
            sess.run(self.op_init)
            self.trained_epochs = 0

        # Training.
        if self.selecting_mode and self.selecting_method == 'masking': self.worder = []
        scores = []
        losses_train = []
        #grads = []
        losses_val = []
        loss_train = 0.0
        indices = collections.deque()
        num_steps = self.num_epochs * int(train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:      
                indices.clear()                                         
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_targets = train_data[idx,:,:], train_targets[idx,]
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices
            if type(batch_targets) is not np.ndarray:
                batch_targets = batch_targets.toarray()

            feed_dict = {self.ph_data: batch_data, self.ph_targets: batch_targets, self.ph_rate_fc: self.rate_fc}
            
            if self.D is not None:
                batch_covariates = train_cova[idx,:]
                if type(batch_covariates) is not np.ndarray:
                    batch_covariates = batch_covariates.toarray()
                feed_dict[self.ph_covariates] = batch_covariates
            if self.selecting_mode:
                feed_dict[self.ph_rate_input] = self.rate_input
                            
            _ = sess.run(self.op_train, feed_dict) 
                        
            feed_dict[self.ph_rate_fc] = 0
            if self.selecting_mode:
                feed_dict[self.ph_rate_input] = 0
                
            batch_loss = sess.run(self.op_loss, feed_dict)
            loss_train += batch_loss
            #grads.append(learning_rate_grad) 
            
            # Evaluation of the model.
            if step % int(train_data.shape[0] / self.batch_size) == 0 or step == num_steps:
                                                    
                self.trained_epochs += 1             
                print('epoch {:} :'.format(int(self.trained_epochs)))
                print('  training_loss = {:.2e}'.format(loss_train))
                if self.selecting_method != 'masking':
                    string, _, r2, loss_val = self.evaluate(val_data, val_targets, val_cova, sess)
                    scores.append(r2)
                    losses_val.append(loss_val)
                    print('  validation {}'.format(string))
                    print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))   
                    # Early stopping
                    if self.selecting_mode and self.selecting_method == 'dropout':
                        if self.trained_epochs > 5 and self.trained_epochs % 5 == 0 and np.mean(losses_val[-10:-5]) < np.mean(losses_val[-5:]):
                            print(' Training stopped for the mean validation loss starts to increase.')   
                            losses_train.append(loss_train)
                            self.op_saver.save(sess, path, global_step=(int(train_data.shape[0] / self.batch_size))*self.trained_epochs)
                            break
                    if not self.selecting_mode:
                        if self.trained_epochs > 2 and np.mean(losses_val[-4:-2]) < np.mean(losses_val[-2:]):
                            losses_train.append(loss_train)
                            self.op_saver.save(sess, path, global_step=(int(train_data.shape[0] / self.batch_size))*self.trained_epochs)
                            print(' Training stopped for the validation loss starts to increase.')    
                            break
                
                losses_train.append(loss_train)
                loss_train = 0.0
                    
                # Summaries for TensorBoard.
#                summary = tf.Summary()
#                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
#                summary.value.add(tag='validation/r2', simple_value=r2)
#                summary.value.add(tag='validation/loss', simple_value=loss_val)
#                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=(int(train_data.shape[0] / self.batch_size))*self.trained_epochs)

                # Plot the mask layer weights
                if self.selecting_mode and self.selecting_method == 'masking':
                    w = sess.run(self.graph.get_tensor_by_name('mask/weights:0'))                    
                    plt.ioff()
                    fig = plt.figure()                    
                    plt.scatter(range(w.shape[1]),w[0,])
                    plt.ylim((-0.05, 1))
                    plt.title('epoch: {}'.format(int(self.trained_epochs)))
                    p = PdfPages('Mask_weights.pdf')
                    p.savefig(fig)
                    p.close()
                    plt.close(fig)                      
                    
        if self.selecting_method != 'masking': 
            print('validation score: peak = {:.2f}, mean = {:.2f}'.format(max(scores), np.mean(scores[-5:])))
            print('validation loss: min = {:.2f}, mean = {:.2f}'.format(min(losses_val), np.mean(losses_val[-5:])))
        #writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        return losses_train, losses_val, scores, t_step#, grads

    def get_var(self, name, sess = None):
        sess = self._get_session(sess)
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val
    
    def get_activation(self, stimuli):
        """Get the activation of GConv block of input stimuli"""
        sess = self._get_session(sess = None)
        _, M, F = self.op_gconv_output.get_shape()
        activations = np.empty((stimuli.shape[0], M, F))
        
        for begin in range(0, stimuli.shape[0], self.batch_size):
            end = begin + self.batch_size
            end = min([end, stimuli.shape[0]])
            
            batch_data = np.zeros((self.batch_size, *stimuli.shape[1:]))
            tmp_data = stimuli[begin:end,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices                          
            if stimuli.shape[0] < begin + self.batch_size:
                ind = list(range(end-begin))*int(self.batch_size/(end-begin)+1)
                ind = ind[:self.batch_size]
                batch_data = tmp_data[ind,]
            else:
                batch_data = tmp_data    
            
            feed_dict = {self.ph_data: batch_data, self.ph_rate_fc: 0}
            if self.selecting_mode:
                feed_dict[self.ph_rate_input] = 0
            batch_act = sess.run(self.op_gconv_output, feed_dict)           
            activations[begin:end,:] = batch_act[:end-begin,:]

        sess.close()
        return activations
    

    # Methods to construct the computational graph.    
    def build_graph(self, L_0, C, D, H):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, [self.batch_size, L_0, H+1], 'data')
                self.ph_targets = tf.placeholder(tf.float32, [self.batch_size, C], 'targets')
                if D is not None:
                    self.ph_covariates = tf.placeholder(tf.float32, [self.batch_size, D], 'covariates')
                self.ph_rate_fc = tf.placeholder(tf.float32, (), 'rate_fc')
                
            # Model.
            if self.selecting_mode:
                self.ph_rate_input = tf.placeholder(tf.float32, (), 'rate_input')
                self.op_select_output = self.selecting_block(self.ph_data, self.selecting_method, self.ph_rate_input)    
                self.op_gconv_output = self.gconv_block(self.op_select_output)
            else:
                self.op_gconv_output = self.gconv_block(self.ph_data)          
            N, M, F = self.op_gconv_output.get_shape()
            gconv_output_flat = tf.reshape(self.op_gconv_output, [int(N), int(M*F)])
            if D is not None:
                fc_input = tf.concat([gconv_output_flat, self.ph_covariates], 1)      
            else:
                fc_input = gconv_output_flat 
            self.op_output = self.fc_block(fc_input, self.M, rate = self.ph_rate_fc)
            self.op_loss = self.loss(self.op_output, self.ph_targets, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = tf.identity(self.op_output)

            # Initialize variables, i.e. weights and biases.b
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            #self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=1)
        
        self.graph.finalize()
    

    def loss(self, output, targets, regularization):
        """Adds to the inference model the layers required to generate loss."""  
        with tf.name_scope('loss'):
            with tf.name_scope('mse'):
                if self.selecting_mode:   
                    string = 'mask/weights:0' if self.selecting_method == 'masking' \
                        else 'dropout/input_node_dropout/Floor:0'
                    w = self.graph.get_tensor_by_name(string)
                    w = tf.slice(w, [0,0,0], [1, output.shape[1], 1]) # remove the fake nodes due to pooling are not taken 
                    w = tf.squeeze(w)
                    mse = tf.nn.l2_loss(tf.multiply(output - targets, 1 - w))
                    if self.selecting_method == 'masking':
                        mse /= self.batch_size
                        mse += self.lambda1 * tf.reduce_sum(tf.abs(w))
                    if self.selecting_method == 'dropout':
                        mse = tf.cond(tf.equal(mse, tf.constant(0.0)), \
                                  lambda: tf.nn.l2_loss(output - targets), \
                                  lambda: mse) # in testing mode, the input dropout is deactivated
                        mse /= self.batch_size
                else:
                    mse = tf.nn.l2_loss(output - targets)/self.batch_size
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = mse + regularization
            
#            # Summaries for TensorBoard.
#            tf.summary.scalar('loss/mse', mse)
#            tf.summary.scalar('loss/regularization', regularization)
#            tf.summary.scalar('loss/total', loss)
#            with tf.name_scope('averages'):
#                averages = tf.train.ExponentialMovingAverage(0.9)
#                op_averages = averages.apply([mse, regularization, loss])
#                tf.summary.scalar('loss/avg/mse', averages.average(mse))
#                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
#                tf.summary.scalar('loss/avg/total', averages.average(loss))
#                with tf.control_dependencies([op_averages]):
#                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss

 
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            #tf.summary.scalar('learning_rate', learning_rate)
            
            # Optimizer.
            if self.selecting_mode and self.selecting_method == 'dropout':
                print("Gradient descent optimizer is being used in current selecting mode.")
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif momentum != 0:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)  
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate)
                print("Adam optimizer is being used.")
                
            grads = optimizer.compute_gradients(loss)   
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
    
            # Histograms.
#            for grad, var in grads:
#                if grad is None:
#                    print('warning: {} has no gradient'.format(var.op.name))
#                else:
#                    tf.summary.histogram(var.op.name + '/gradients', grad)
                    
            # The op returns the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train, grads


    # Helper methods.
    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        K.set_session(sess)
        return sess       

    def _weight_variable(self, shape, regularization=False):
        initial = tf.truncated_normal_initializer(0, 0.05)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        #tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=False):
        initial = tf.constant_initializer(0.05)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        #tf.summary.histogram(var.op.name, var)
        return var



class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hops.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    The following are hyper-parameters of fully connected layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of outputs.
           list, which length is equal to the number of fc layers.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.
        
    Training parameters:
        num_epochs:    Maximal number of training epochs.
        learning_rate: learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases. No regularization with 0.
        rate_fc:        Dropout rate (fc layers): probability to keep hidden neurons. No dropout with 0.
        batch_size:     Batch size. Not necessarily to divide evenly into the dataset sizes.
                        Recommend to set with a small value in selecting mode, especially when dropout method is used.
        
    Directories:
        dir_name: Name for directories (model parameters).
        
        
    The following are newly added settings by this version.
    D (optional): Number of covariates, required when external covariates are used. 
                  The external features will be input to the FC block,
                  together with all the output vertex features from the GConv block.
                  When D is not none, train_cova and val_cova are required by the fit function,
                  covariates is required by the evaluate function.
    H : time lag.
    selecting_mode: if True, build selection network: 
                        selecting_method: 'dropout', 'masking':
                            when dropout method is used:
                                rate_input: Dropout rate (input layer).
                            when masking method is used:
                                lambda1: Tuning parameter of L1 loss of mask weights.
                        P: The expected number of selected sensor.
    
                    else if False, build prediction network. 
    """
    def __init__(self, L, F, K, p, M, D = None, H = 0, 
                selecting_mode = False, selecting_method = 'dropout', rate_input = 0.05, lambda1 = 0.5, P = None,
                filter='chebyshev5', brelu='b2relu', pool='apool1',
                num_epochs=20, learning_rate=0.1, decay_rate=1, decay_steps=10, momentum=0,
                regularization=0, rate_fc=0, batch_size=30, dir_name=''):
        super().__init__()
        
        # Verify the consistency w.r.t. the number of layers.
        assert len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1), "Pooling sizes should be at least 1 (no pooling)."
        p_log2 = np.log2(p)
        assert np.all(np.mod(p_log2, 1) == 0), "Pooling sizes should be powers of 2."
        #assert len(L) >= 1 + np.sum(p_log2), "Not enough coarsening levels in L for pool sizes."
        
        # Keep the useful Laplacians only. May be zero.
        L_0 = L[0].shape[0]
        j = 0
        self.L = [L[j]]
        for pp in p[:-1]:  
            j += int(np.log2(pp))
            self.L.append(L[j])
        L = self.L
        
        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        nparam = L_0
        print('NN architecture')
        print('  input: L_0 = {}'.format(L_0))
        if selecting_mode and selecting_method == 'masking':
            print('  mask layer: L_0 = {}'.format(L_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: L_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))              
            F_last = F[i-1] if i > 0 else H + 1
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            nparam += F_last*F[i]*K[i]
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
                nparam += F[i]
            elif brelu == 'b2relu':
                print('    biases: L_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))
                nparam += L[i].shape[0]*F[i]
                
        for i in range(Nfc):
            name = 'output' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: L_{} = {}'.format(Ngconv+i+1, M[i]))
            D0 = 0 if D is None else D
            M_last = M[i-1] if i > 0 else L_0 + D0 if Ngconv == 0 else L[-1].shape[0] * F[-1] //p[-1] +D0
            print('    weights: L_{} * L_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            print('    biases: L_{} = {}'.format(Ngconv+i+1, M[i]))
            nparam += M_last * M[i] + M[i]
            
        if selecting_mode and selecting_method == 'masking':
            nparam += L_0
        print('  Total number of trainable parameters: {}'.format(nparam))    
        
        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M, self.D, self.H, self.P = L, F, K, p, M, D, H, P
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.rate_input, self.rate_fc, self.lambda1 \
                = regularization, rate_input, rate_fc, lambda1 
        self.flag_regularization = False if self.regularization == 0 else True 
        self.batch_size = batch_size
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        self.selecting_mode = selecting_mode
        self.selecting_method = selecting_method
        self.trained_epochs = 0      
        
        # Build the computational graph.
        self.build_graph(L_0, M[-1], D, H)

    def chebyshev2(self, x, L, Fout, K, regularization=False):
        """
        Filtering with Chebyshev interpolation
        Implementation: numpy.
        
        Data: x of size N x M x F
            N: number of signals
            M: number of vertices
            F: number of features per signal per vertex
        """
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        #L = graph.rescale_L(L, lmax=2)
        L = graph.rescale_L(L)
        # Transform to Chebyshev basis
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        def chebyshev(x):
            return graph.chebyshev(L, x, K)
        x = tf.py_func(chebyshev, [x], [tf.float32])[0]  # K x M x Fin*N
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature.
        W = self._weight_variable([Fin*K, Fout], regularization)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def chebyshev5(self, x, L, Fout, K, regularization=False):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        L = scipy.sparse.csr_matrix(L)
        #L = graph.rescale_L(L, lmax=2)
        L = graph.rescale_L(L)
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        L = tf.sparse_reorder(L)
        # Transform to Chebyshev basis
        x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x M x Fin*N
        def concat(x, x_):
            x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            return tf.concat([x, x_], axis=0)  # K x M x Fin*N
        if K > 1:
            x1 = tf.sparse_tensor_dense_matmul(L, x0)
            x = concat(x, x1)
        for k in range(2, K):
            x2 = 2 * tf.sparse_tensor_dense_matmul(L, x1) - x0  # M x Fin*N
            x = concat(x, x2)
            x0, x1 = x1, x2
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout
    
    def b1relu(self, x, regularization=False):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization)
        return tf.nn.elu(x + b)

    def b2relu(self, x, regularization=False):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization)
        return tf.nn.elu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, activation='relu', regularization = True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization)
        b = self._bias_variable([Mout], regularization)
        x = tf.matmul(x, W) + b
        if activation == 'relu':
            return tf.nn.relu(x)
        if activation == 'leaky_relu':
            return tf.nn.leaky_relu(x)
        if activation == 'none':
            return x
    

    # Inference.   
    def selecting_block(self, x, method, rate = 0.05):
                
        if len(x.shape) == 2:
            x = tf.expand_dims(x, 2)  # N(depth)x M(row)x Fin(=1,col)  
        
        N, M, Fin = x.get_shape()
        if method == 'dropout':
            with tf.variable_scope('dropout'):
                x = tf.nn.dropout(x, rate = rate, noise_shape = [1, M, 1], name='input_node_dropout')
                x /= (1 - rate)
        elif method == 'masking':
            with tf.variable_scope('mask'):
                with tf.name_scope('obsvervations'):
                    initial = tf.initializers.random_uniform(0.6,0.8)
                    w = tf.get_variable('weights', [1, M, 1], tf.float32, initializer=initial, constraint=lambda x: tf.clip_by_value(x, 0, 1))                        
                    #tf.summary.histogram(w.op.name, w)                           
                    x = tf.multiply(x, w) 
        else:
            raise NameError("Only 2 selecting methods are available: 'dropout', 'masking'.")
        return x    
    
    
    def gconv_block(self, x):
                                                      
        if len(x.shape) == 2:
            x = tf.expand_dims(x, 2)  # N(depth)x M(row)x Fin(=1,col)  
            
        # Graph convolutional layers.     
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    x = self.filter(x, self.L[i], self.F[i], self.K[i], self.flag_regularization)
                with tf.name_scope('bias_relu', self.flag_regularization):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])       
        return x
    
    
    def fc_block(self, x, M, rate = 0):
        
        # Fully connected hidden layers.
        for i,m in enumerate(M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, m, regularization = self.flag_regularization, activation='leaky_relu')
                x = tf.nn.dropout(x, rate = 0)
                
        # Output layer.
        with tf.variable_scope('output'):
            x = self.fc(x, M[-1], activation='none')
        return x
