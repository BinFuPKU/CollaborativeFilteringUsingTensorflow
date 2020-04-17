import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
# from tqdm import tqdm
import gc

import sys
sys.path.append("../../metrics/")
from rating import evaluate

class SVD(object):
    def __init__(self, n_users, n_items, eval_metrics=['rmse','mae'],
                 range_of_ratings=(.5,5), reg=0.02, n_factors=10, batch_size=500,
                 max_iter=50, lr=.1,
                 init_mean=0.0, init_stddev=0.1,
                 device='CPU'):
        self.__n_users, self.__n_items, self.__eval_metrics = n_users, n_items, eval_metrics

        self.__range_of_ratings, self.__reg, self.__n_factors, self.__batch_size = range_of_ratings, reg, n_factors, batch_size
        self.__max_iter, self.__lr,  = max_iter, lr
        self.__init_mean, self.__init_stddev = init_mean, init_stddev

        self.__device = device
        self.__DEVICES = [x.name for x in list_local_devices() if x.device_type == device]

        self.__user_embed = tf.get_variable(shape=[self.__n_users,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                          name='user_embed')
        self.__kernel = tf.get_variable(shape=[self.__n_factors, self.__n_factors],
                                            initializer=tf.truncated_normal_initializer(mean=self.__init_mean, stddev=self.__init_stddev),
                                            name='kernel')
        self.__item_embed = tf.get_variable(shape=[self.__n_items,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='item_embed')
        # data input:
        self.__useritem_placeholder = tf.placeholder(tf.int32, shape=[None, 2])
        self.__rating_placeholder = tf.placeholder(tf.float32, shape=[None])

        # property
        self.__reg_loss__
        self.__embed_loss__
        self.__loss
        self.__predict
        self.__optimize__
        self.__eval

        self.__sess = None


    @property
    def __reg_loss__(self):
        reg_loss_ = tf.nn.l2_loss(tf.nn.embedding_lookup(self.__user_embed, self.__useritem_placeholder[:, 0]))
        reg_loss_ += tf.nn.l2_loss(tf.nn.embedding_lookup(self.__item_embed, self.__useritem_placeholder[:, 1]))
        return  self.__reg * reg_loss_

    @property
    def __embed_loss__(self):
        return tf.nn.l2_loss(self.__predict - self.__rating_placeholder)

    @property
    def __loss(self):
        return self.__embed_loss__  + self.__reg_loss__

    @property
    def __predict(self):
        # user embedding (N, K)
        user_embed = tf.nn.embedding_lookup(self.__user_embed, self.__useritem_placeholder[:, 0], name="user_embed_")
        # item embedding (N, K)
        items_embed = tf.nn.embedding_lookup(self.__item_embed, self.__useritem_placeholder[:, 1], name="items_embed_")
        rating_ = tf.reduce_sum(tf.matmul(user_embed, self.__kernel) * items_embed, reduction_indices=1)
        return rating_

    @property
    def __optimize__(self):
        gds = []
        gds.append(tf.train.AdagradOptimizer(self.__lr).minimize(self.__loss, var_list=[self.__user_embed, self.__kernel, self.__item_embed]))
        with tf.control_dependencies(gds):
            return gds + [self.__user_embed, self.__kernel, self.__item_embed]

    def __eval(self, tst_tuple):
        ratings_ = self.__sess.run(tf.clip_by_value(self.__predict, self.__range_of_ratings[0], self.__range_of_ratings[1]),
                                   {self.__useritem_placeholder: tst_tuple[:,:-1]})
        return evaluate(tst_tuple[:,-1], ratings_, self.__eval_metrics)


    def train(self, fold, tra_tuple, tst_tuple, sampler):
        scores = None
        with tf.device(self.__DEVICES[0]):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.__sess = tf.Session(config=config)
            train_op = (self.__optimize__, self.__loss)
            self.__sess.run(tf.global_variables_initializer())

            n_batches = int(len(tra_tuple)/self.__batch_size)

            # sample all users
            for iter in range(self.__max_iter):
                losses = []
                for x in range(n_batches): # tqdm(range(n_batches), desc="\tOptimizing")
                    useritemrating_tuples_batch = sampler.next_batch()
                    _, loss = self.__sess.run((train_op),{self.__useritem_placeholder: useritemrating_tuples_batch[:,:-1],
                                                   self.__rating_placeholder: useritemrating_tuples_batch[:,-1]})
                    losses.append(loss)
                aveloss = np.mean(losses)

                scores = self.__eval(tst_tuple)

                print("fold=%d iter=%2d: " % (fold, iter + 1),
                      "TraLoss=%.4f lr=%.4f" % (aveloss, self.__lr),
                      '\tTst:' + ' '.join(
                          [eval_metric + '=%.4f' % (score) for eval_metric, score in zip(self.__eval_metrics, scores)]))

                self.__lr *= .98
                gc.collect()
        return scores

    def close(self):
        self.__sess.close()