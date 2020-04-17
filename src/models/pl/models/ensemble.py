import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
# from tqdm import tqdm

import gc

import sys
sys.path.append("../../metrics/")
from ranking import evaluateCV, evaluateLOOV

class Ensemble(object):
    def __init__(self, n_users, n_items,
                 kensemble = 3,
                 topN=5,
                 split_method='cv', eval_metrics=['pre','recall','mrr', 'ndcg'],
                 reg=0.1, n_factors=20, batch_size=100,
                 max_iter=50, lr=0.1,
                 init_mean=0.0, init_stddev=0.1,
                 device='CPU'):
        # parameters
        self.__n_users, self.__n_items, self.__topN = n_users, n_items, topN
        self.__split_method, self.__eval_metrics = split_method, eval_metrics
        self.__reg, self.__n_factors, self.__batch_size = reg, n_factors, batch_size
        self.__max_iter, self.__lr = max_iter, lr
        self.__init_mean, self.__init_stddev = init_mean, init_stddev

        self.__device = device
        self.__DEVICES = [x.name for x in list_local_devices() if x.device_type == device]

        self.kensemble = kensemble
        self.__user_embeds = tf.get_variable(shape=[self.kensemble, self.__n_users, self.__n_factors],
                        initializer=tf.truncated_normal_initializer(mean=self.__init_mean, stddev=self.__init_stddev),
                        name='user_embeds')
        self.__item_embeds = tf.get_variable(shape=[self.kensemble, self.__n_items,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='item_embeds')
        self.__h_embeds = tf.get_variable(shape=[self.kensemble,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='h_embeds')

        # data input:
        self.__uij = tf.placeholder(tf.int32, shape=[None, 3])

        # evaluate
        self.__users_placeholder = tf.placeholder(tf.int32, shape=[None])

        # property
        self.__reg_loss__
        self.__embed_loss__
        self.__loss
        self.__predict__
        self.__optimize__

        self.__sess = None
        self.__sampler = None

    def __reg_k_loss__(self, k):
        reg_k_loss_ = tf.nn.l2_loss(tf.nn.embedding_lookup(self.__user_embeds[k,:,:], self.__uij[:, 0]))
        reg_k_loss_ += tf.nn.l2_loss(tf.nn.embedding_lookup(self.__item_embeds[k,:,:], self.__uij[:, 1:]))
        return reg_k_loss_

    @property
    def __reg_loss__(self):
        reg_loss_ = 0
        for i in range(self.kensemble):
            reg_loss_ += self.__reg_k_loss__(i)
        reg_loss_ += tf.nn.l2_loss(self.__h_embeds)
        return self.__reg * reg_loss_

    def __embed_score_a__(self, k):
        # user embedding (N, K)
        user_embed = tf.nn.embedding_lookup(self.__user_embeds[k,:,:], self.__uij[:, 0])
        # positive item embedding (N, K)
        posItem_embed = tf.nn.embedding_lookup(self.__item_embeds[k,:,:], self.__uij[:, 1])
        # negative item embedding (N, K)
        negItem_embed = tf.nn.embedding_lookup(self.__item_embeds[k,:,:], self.__uij[:, 2])

        # n * k
        ui = user_embed * posItem_embed
        # n * k
        uj = user_embed * negItem_embed
        # n
        ui_a = tf.exp(tf.matmul(ui, tf.expand_dims(tf.transpose(self.__h_embeds[k,:]), 1)))
        # n
        uj_a = tf.exp(tf.matmul(uj, tf.expand_dims(tf.transpose(self.__h_embeds[k,:]), 1)))

        # n
        ui_rating = tf.reduce_sum(ui, reduction_indices=-1) * ui_a
        # n
        uj_rating = tf.reduce_sum(uj, reduction_indices=-1) * uj_a

        return ui_rating, uj_rating, ui_a, uj_a


    @property
    def __embed_loss__(self):
        score_a = []
        ui_a_base, uj_a_base = 0, 0
        for i in range(self.kensemble):
            x1, x2, x3, x4 = self.__embed_score_a__(i)
            score_a.append([x1, x2, x3, x4])
            ui_a_base += score_a[-1][-2]
            uj_a_base += score_a[-1][-1]
        ui_rating, uj_rating = 0, 0
        for i in range(self.kensemble):
            ui_rating += score_a[i][0] / ui_a_base
            uj_rating += score_a[i][1] / uj_a_base
        embed_loss_ = tf.reduce_sum(-tf.log(tf.sigmoid(ui_rating - uj_rating)), name="embed_loss_")
        return embed_loss_

    @property
    def __loss(self):
        return self.__embed_loss__  + self.__reg_loss__

    def __predict_k__(self, k):
        # n * 1 * k
        tst_user_embed = tf.expand_dims(tf.nn.embedding_lookup(self.__user_embeds[k,:,:], self.__users_placeholder), 1)
        #  *   1 * m * k = n * m * k
        ui = tst_user_embed * tf.expand_dims(self.__item_embeds[k,:,:], 0)

        # n * m
        ui_score = tf.reduce_sum(ui, reduction_indices=-1)
        # n * m
        ui_a = tf.reduce_sum(ui * tf.expand_dims(self.__h_embeds[k,:], 0), reduction_indices=-1)
        return ui_score, ui_a

    @property
    def __predict__(self):
        score_a = []
        ui_a_base = 0
        for i in range(self.kensemble):
            x1, x2 = self.__predict_k__(i)
            score_a.append([x1, tf.exp(x2)])
            ui_a_base += score_a[-1][-1]
        ui_rating, uj_rating = 0, 0
        for i in range(self.kensemble):
            score_a[i][-1] /= ui_a_base
            ui_rating += score_a[i][0] * score_a[i][-1]
        return ui_rating


    @property
    def __optimize__(self):
        gds = []
        gds.append(tf.train.AdagradOptimizer(self.__lr).minimize(self.__loss, var_list=[self.__user_embeds, self.__item_embeds, self.__h_embeds]))
        with tf.control_dependencies(gds):
            return gds + [self.__user_embeds, self.__item_embeds, self.__h_embeds]

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

    def __eval(self, yss_true, yss_pred):
        if self.__split_method=='cv':
            return evaluateCV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        elif self.__split_method=='loov':
            return evaluateLOOV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        else:
            return None

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
            train_op = (self.__optimize__, self.__loss) # must before the initializer
            self.__sess = tf.Session(config=config)
            self.__sess.run(tf.global_variables_initializer())

            n_batches = int(len(tra_tuple)/self.__batch_size)

            # sample all users
            for iter in range(self.__max_iter):
                losses = []
                for _ in range(n_batches):
                    uij_batch = self.__sampler.next_batch()
                    _, loss = self.__sess.run(train_op, {self.__uij: uij_batch})

                    losses.append(loss)

                aveloss = np.mean(losses)

                yss_pred = self.__recommend(test_users, tstintra_set)
                scores = self.__eval(yss_true, yss_pred)
                print("%s_fold=%d iter=%2d: " % (self.__split_method, fold, iter + 1),
                      "TraLoss=%.4f lr=%.4f" % (aveloss, self.__lr),
                      '\tTst@' + str(self.__topN) + ':' + ' '.join(
                          [eval_metric + '=%.4f' % (score) for eval_metric, score in zip(self.__eval_metrics, scores)]))

                self.__lr *= .98
                gc.collect()
        # topNs = [5,10,20,50,100,200,500,1000]
        # self.__topN=topNs[-1]
        # yss_pred = self.__recommend(test_users, tstintra_set)
        # for topN in topNs:
        #     self.__topN = topN
        #     scores = self.__eval(yss_true, yss_pred)
        #     print("%s_fold=%d: " % (self.__split_method, fold),
        #       '\tTst@' + str(self.__topN) + ':' + ' '.join(
        #           [eval_metric + '=%.4f' % (score) for eval_metric, score in zip(self.__eval_metrics, scores)]))
        return scores

    def close(self):
        self.__sess.close()
