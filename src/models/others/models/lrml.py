
import numpy as np
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
import gc

import sys
sys.path.append("../../metrics/")
from ranking import evaluateCV, evaluateLOOV

class LRML(object):

    def __init__(self, n_users, n_items, topN=5,
                 split_method='cv', eval_metrics=['pre','recall','mrr', 'ndcg'],
                 n_memory=20, margin=.2, clip_norm=1.0,
                 n_factors=50, batch_size=100,
                 max_iter=50, lr=0.01,
                 init_mean=0.0, init_stddev=0.01,
                 device='CPU'):
        # parameters
        self.__n_users, self.__n_items, self.__topN = n_users, n_items, topN
        self.__split_method, self.__eval_metrics = split_method, eval_metrics
        self.__n_memory, self.__margin, self.__clip_norm = n_memory, margin, clip_norm
        self.__n_factors, self.__batch_size = n_factors, batch_size
        self.__max_iter, self.__lr = max_iter, lr
        self.__init_mean, self.__init_stddev = init_mean, init_stddev
        self.__device = device
        self.__DEVICES = [x.name for x in list_local_devices() if x.device_type == device]

        self.__user_embed = tf.get_variable(shape=[self.__n_users,self.__n_factors],
                                initializer=tf.random_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev, dtype=tf.float32),
                                            name='user_embed')
        self.__item_embed = tf.get_variable(shape=[self.__n_items,self.__n_factors],
                                initializer=tf.random_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev, dtype=tf.float32),
                                            name='item_embed')
        self.__useritemKey = tf.get_variable(shape=[self.__n_factors, self.__n_memory],
                                initializer=tf.random_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev, dtype=tf.float32),
                                            name='useritemKey')
        self.__memory = tf.get_variable(shape=[self.__n_memory, self.__n_factors],
                                initializer=tf.random_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev, dtype=tf.float32),
                                            name='memory')

        # data input:
        self.__pos_useritem_placeholder = tf.placeholder(tf.int32, shape=[None, 2])
        self.__neg_useritem_placeholder = tf.placeholder(tf.int32, shape=[None, 2])

        # evaluate
        self.__users_placeholder = tf.placeholder(tf.int32, shape=[None])

        # property
        self.__embed_loss__
        self.__loss
        self.__predict__
        self.__clip_by_norm_op__
        self.__optimize__

        self.__sess = None

    @property
    def __embed_loss__(self):
        # positive user embedding (N, K)
        pos_user_embed = tf.nn.embedding_lookup(self.__user_embed, self.__pos_useritem_placeholder[:, 0], name="pos_user_embed_")
        # positive item embedding (N, K)
        pos_item_embed = tf.nn.embedding_lookup(self.__item_embed, self.__pos_useritem_placeholder[:, 1], name="pos_item_embed_")

        # netative user embedding (N, K)
        neg_user_embed = tf.nn.embedding_lookup(self.__user_embed, self.__neg_useritem_placeholder[:, 0], name="neg_user_embed_")
        # positive item embedding (N, K)
        neg_item_embed = tf.nn.embedding_lookup(self.__item_embed, self.__neg_useritem_placeholder[:, 1], name="neg_item_embed_")

        # use pos useritem_key for both pos and neg user-item pair
        useritem_embed = tf.multiply(pos_user_embed, pos_item_embed, name='useritem_embed')
        attention = tf.nn.softmax(tf.matmul(useritem_embed, self.__useritemKey), name='attention')
        relation = tf.matmul(attention, self.__memory, name='relation')

        pos_score = tf.reduce_sum(tf.square(pos_user_embed + relation - pos_item_embed), reduction_indices=-1, name='pos_score')
        neg_score = tf.reduce_sum(tf.square(neg_user_embed + relation - neg_item_embed), reduction_indices=-1, name='neg_score')

        embed_loss = tf.reduce_sum(tf.nn.relu(pos_score + self.__margin - neg_score), name="embed_loss_")
        return embed_loss

    @property
    def __loss(self):
        return self.__embed_loss__

    @property
    def __predict__(self):
        # (U, 1, K)
        tst_user_embed = tf.expand_dims(tf.nn.embedding_lookup(self.__user_embed, self.__users_placeholder), 1, name="tst_user_embed_")
        # (1, I, K)
        all_item_embed = tf.expand_dims(self.__item_embed, 0, name='all_item_embed')

        # use pos useritem_key for both pos and neg user-item pair
        # (U, I, K)
        useritem_embed_ = tf.multiply(tst_user_embed, all_item_embed, name='useritem_embed_')
        # (U, I, 1, K) * (K, N).T
        attention_ = tf.nn.softmax(tf.multiply(tf.expand_dims(useritem_embed_, 2),
                                              tf.transpose(self.__useritemKey)), name='attention_')
        # (U, I, N, K) * (N, K) ->  (U, I, K)
        relation_ = tf.reduce_sum(tf.multiply(attention_, self.__memory), reduction_indices=2, name='relation_')
        # (U, 1, K) + (U, I, K) - (1, I, K) -> (U, I)
        predicts = - tf.reduce_sum(tf.square(tst_user_embed + relation_ - all_item_embed), reduction_indices=-1, name='predicts')

        return predicts

    @property
    def __clip_by_norm_op__(self):
        return [tf.assign(self.__user_embed, tf.clip_by_norm(self.__user_embed, self.__clip_norm, axes=[1])),
                tf.assign(self.__item_embed, tf.clip_by_norm(self.__item_embed, self.__clip_norm, axes=[1]))]

    @property
    def __optimize__(self):
        gds = []
        gds.append(tf.train.AdagradOptimizer(self.__lr).minimize(self.__loss, var_list=[self.__user_embed, self.__item_embed]))
        with tf.control_dependencies(gds):
            return gds + [self.__clip_by_norm_op__]

    def __recommend(self, test_users, tstintra_set):

        yss_pred = []
        # batch
        start=0
        while start<len(test_users):
            end = min(start+self.__batch_size, len(test_users))
            itemset_maxsize = max([len(itemset) for itemset in tstintra_set[start: end]])
            yss_pred_batch = self.__sess.run(tf.nn.top_k(self.__predict__, itemset_maxsize + self.__topN),
                                        {self.__users_placeholder: test_users[start: end]})[1]
            # filter out the rated items
            for ind in range(yss_pred_batch.shape[0]):
                yss_pred.append([])
                u = start + ind
                for i in yss_pred_batch[ind,:]:
                    if i not in tstintra_set[u]:
                        yss_pred[-1].append(i)
                    if len(yss_pred[-1]) >= self.__topN:
                        break
            start += self.__batch_size
            gc.collect()
        return yss_pred


    def __eval(self, yss_true, yss_pred):
        if self.__split_method=='cv':
            return evaluateCV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        elif self.__split_method=='loov':
            return evaluateLOOV(yss_true, yss_pred, self.__eval_metrics, self.__topN)
        else:
            return None

    def train(self, fold, trasR, tstsR, sampler):

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

            n_batches = 5 * int(len(tra_tuple)/self.__batch_size)

            # sample all users
            for iter in range(self.__max_iter):
                losses = []
                for _ in range(n_batches):
                    pos_useritem_batch, neg_useritem_batch = sampler.next_batch()
                    _, loss = self.__sess.run(train_op, {self.__pos_useritem_placeholder: pos_useritem_batch,
                                                   self.__neg_useritem_placeholder: neg_useritem_batch})

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
        return scores

    def close(self):
        self.__sess.close()
