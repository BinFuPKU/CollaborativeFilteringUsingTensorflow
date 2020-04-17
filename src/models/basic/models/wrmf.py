import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
# from tqdm import tqdm
import gc

import sys
sys.path.append("../../metrics/")
from ranking import evaluateCV, evaluateLOOV

class WRMF(object):
    def __init__(self, n_users, n_items, topN=10,
                 split_method='cv', eval_metrics=['pre','recall', 'map', 'mrr', 'ndcg'],
                 weight=1, reg=0.02, n_factors=10, batch_size=500,
                 max_iter=50, lr=.1,
                 init_mean=0.0, init_stddev=0.1,
                 device='CPU'):
        self.__n_users, self.__n_items, self.__topN = n_users, n_items, topN
        self.__split_method, self.__eval_metrics = split_method, eval_metrics
        self.__weight, self.__reg, self.__n_factors, self.__batch_size = weight, reg, n_factors, batch_size
        self.__max_iter, self.__lr,  = max_iter, lr
        self.__init_mean, self.__init_stddev = init_mean, init_stddev

        self.__device = device
        self.__DEVICES = [x.name for x in list_local_devices() if x.device_type == device]

        self.__user_embed = tf.get_variable(shape=[self.__n_users,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='user_embed')
        self.__item_embed = tf.get_variable(shape=[self.__n_items,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='item_embed')
        # data input:
        self.__useritem_placeholder = tf.placeholder(tf.int32, shape=[None, 2])
        self.__rating_placeholder = tf.placeholder(tf.float32, shape=[None])

        # evaluate
        self.__users_placeholder = tf.placeholder(tf.int32, shape=[None])

        # property
        self.__reg_loss__
        self.__embed_loss__
        self.__loss
        self.__predict
        self.__predict__
        self.__optimize__

        self.__sess = None
        self.__sampler = None


    @property
    def __reg_loss__(self):
        reg_loss_ = tf.nn.l2_loss(tf.nn.embedding_lookup(self.__user_embed, self.__useritem_placeholder[:, 0]))
        reg_loss_ += tf.nn.l2_loss(tf.nn.embedding_lookup(self.__item_embed, self.__useritem_placeholder[:, 1]))
        return  self.__reg * reg_loss_

    @property
    def __embed_loss__(self):
        return tf.nn.l2_loss((self.__predict - self.__rating_placeholder) * np.sqrt(self.__weight))
                             # (tf.cast(tf.greater(self.__rating_placeholder, 0), tf.float32) * np.sqrt(self.__weight) +
                             #  tf.cast(tf.less_equal(self.__rating_placeholder, 0), tf.float32)))

    @property
    def __loss(self):
        return self.__embed_loss__  + self.__reg_loss__

    @property
    def __predict(self):
        # user embedding (N, K)
        user_embed = tf.nn.embedding_lookup(self.__user_embed, self.__useritem_placeholder[:, 0], name="user_embed_")
        # item embedding (N, K)
        items_embed = tf.nn.embedding_lookup(self.__item_embed, self.__useritem_placeholder[:, 1], name="items_embed_")
        rating_ = tf.reduce_sum(user_embed * items_embed, reduction_indices=1)
        return rating_

    @property
    def __predict__(self):
        tst_user_embed = tf.nn.embedding_lookup(self.__user_embed, self.__users_placeholder, name="tst_user_embed_")
        predicts = tf.matmul(tst_user_embed, tf.transpose(self.__item_embed), name="predicts")
        return predicts

    @property
    def __optimize__(self):
        gds = []
        gds.append(tf.train.AdagradOptimizer(self.__lr).minimize(self.__loss, var_list=[self.__user_embed, self.__item_embed]))
        with tf.control_dependencies(gds):
            return gds + [self.__user_embed, self.__item_embed]

    def __eval(self, yss_true, yss_pred):
        if self.__split_method=='cv':
            return evaluateCV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        elif self.__split_method=='loov':
            return evaluateLOOV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        else:
            return None

    def __recommend(self, test_users, tstintra_set):
        itemset_maxsize = max([len(itemset) for itemset in tstintra_set])
        yss_pred_ = self.__sess.run(tf.nn.top_k(self.__predict__, itemset_maxsize + self.__topN),
                                    {self.__users_placeholder: test_users})[1]
        # filter out the rated items
        yss_pred = []
        for ind in range(len(test_users)):
            yss_pred.append([])
            for y_pred_ in yss_pred_[ind]:
                if y_pred_ not in tstintra_set[ind]:
                    yss_pred[-1].append(y_pred_)
                if len(yss_pred[-1]) >= self.__topN:
                    break
        return yss_pred


    def train(self, fold, trasR, tstsR, sampler):
        self.__sampler = sampler
        # for eval:
        # tst
        test_users = list(set(np.asarray(tstsR.nonzero()[0])))
        yss_true = None
        if self.__split_method=='cv':
            yss_true = [set(tstsR[user].nonzero()[1]) for user in test_users]
        elif self.__split_method=='loov':
            yss_true = [tstsR[user].nonzero()[1][0] for user in test_users]

        # tra
        tstintra_set = [set(trasR[user].nonzero()[1]) for user in test_users]

        # for train:
        tra_tuple = np.array([(user, item) for user, item in np.asarray(trasR.nonzero()).T])  # double

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
                for x in range(n_batches):
                    useritemrating_tuples_batch = sampler.next_batch()
                    _, loss = self.__sess.run((train_op),{self.__useritem_placeholder: useritemrating_tuples_batch[:,:-1],
                                                   self.__rating_placeholder: useritemrating_tuples_batch[:,-1]})
                    losses.append(loss)
                aveloss = np.mean(losses)

                yss_pred = self.__recommend(test_users, tstintra_set)
                scores = self.__eval(yss_true, yss_pred)

                print("fold=%d iter=%2d: " % (fold, iter + 1),
                      "TraLoss=%.4f lr=%.4f" % (aveloss, self.__lr),
                      '\tTst:' + ' '.join(
                          [eval_metric + '=%.4f' % (score) for eval_metric, score in zip(self.__eval_metrics, scores)]))

                self.__lr *= .98
                gc.collect()
        return scores

    def close(self):
        self.__sess.close()