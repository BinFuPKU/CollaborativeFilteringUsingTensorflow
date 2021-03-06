import numpy as np
from scipy.sparse import lil_matrix
import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
import datetime as dt

import gc

import sys
sys.path.append("../../metrics/")
from ranking import evaluateCV, evaluateLOOV

import sys
sys.path.append("../../samplers/")
from sampler_uitj_ranking import Sampler

class CPLR(object):
    def __init__(self, n_users, n_items,
                 topK=50, topN=5,
                 split_method='cv', eval_metrics=['pre','recall','map','mrr', 'ndcg'],
                 alpha=1., beta=1., gamma=1.,
                 reg=0.01, n_factors=20, batch_size=1000,
                 max_iter=50, lr=0.1,
                 init_mean=0.0, init_stddev=0.1,
                 device='CPU'):
        # parameters
        self.__n_users, self.__n_items, self.__topK, self.__topN = n_users, n_items, topK, topN
        self.__split_method, self.__eval_metrics = split_method, eval_metrics
        self.__alpha, self.__beta, self.__gamma = alpha, beta, gamma
        self.__reg, self.__n_factors, self.__batch_size = reg, n_factors, batch_size
        self.__max_iter, self.__lr = max_iter, lr
        self.__init_mean, self.__init_stddev = init_mean, init_stddev

        self.__device = device
        self.__DEVICES = [x.name for x in list_local_devices() if x.device_type == device]

        self.__user_embed = tf.get_variable(shape=[self.__n_users,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='user_embed')
        self.__item_embed = tf.get_variable(shape=[self.__n_items,self.__n_factors],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='item_embed')
        self.__item_bias = tf.get_variable(shape=[self.__n_items],
                                initializer=tf.truncated_normal_initializer(mean=self.__init_mean,stddev=self.__init_stddev),
                                            name='item_bias')
        # data input:
        self.__uitj_placeholder = tf.placeholder(tf.int32, shape=[None, 4])
        self.__coefs_placeholder = tf.placeholder(tf.float32, shape=[None, 2])

        # evaluate
        self.__users_placeholder = tf.placeholder(tf.int32, shape=[None])

        # property
        self.__reg_loss__
        self.__embed_loss__
        self.__loss
        self.__predict__
        self.__optimize__

        self.__simMat = None
        self.__coefMat = None

        self.__sess = None
        self.__sampler = None

    def __calsim__(self, trasR):
        __simMat = (np.dot(trasR, trasR.T)).toarray()
        for ind in range(trasR.shape[0]):
            den = np.linalg.norm(trasR[ind,:].toarray())
            if den>0:
                __simMat[ind,:] = __simMat[ind,:] / den
                __simMat[:,ind] = __simMat[:,ind] / den
            __simMat[ind, ind] = 0
        return __simMat

    def __topk__(self, array):
        for ind in range(array.shape[0]):
            row_sim = np.zeros((array.shape[1]))
            inds = []
            if self.__topK < len(array[ind, :].nonzero()[0]):
                inds = np.argsort(array[ind, :])[-self.__topK:]
            else:
                inds = array[ind, :].nonzero()[0]
            for ind_ in inds:
                row_sim[ind_] = array[ind, ind_]
            array[ind, :] = row_sim
        return array

    def __calcoef__(self, trasR):
        __coefMat = lil_matrix(trasR.shape)
        for user in set(trasR.nonzero()[0]):
            user_predict = np.zeros(self.__n_items)
            for nn_user in self.__simMat[user,:].nonzero()[0]:
                user_predict += self.__simMat[user,nn_user] * trasR[nn_user].toarray()[0]
            __coefMat[user,:] = lil_matrix(user_predict)
        return __coefMat

    @property
    def __reg_loss__(self):
        reg_loss_ =  tf.nn.l2_loss(tf.nn.embedding_lookup(self.__user_embed, self.__uitj_placeholder[:, 0]))
        reg_loss_ += tf.nn.l2_loss(tf.nn.embedding_lookup(self.__item_embed, self.__uitj_placeholder[:, 1:]))
        reg_loss_ += tf.nn.l2_loss(tf.nn.embedding_lookup(self.__item_bias,  self.__uitj_placeholder[:, 1:]))
        return  self.__reg * reg_loss_


    @property
    def __embed_loss__(self):
        # user embedding (N, K)
        u_embed = tf.nn.embedding_lookup(self.__user_embed, self.__uitj_placeholder[:, 0], name="u_embed_")
        # positive item embedding (N, K)
        i_embed = tf.nn.embedding_lookup(self.__item_embed, self.__uitj_placeholder[:, 1], name="i_embed_")
        # collaborative item embedding (N, K)
        t_embed = tf.nn.embedding_lookup(self.__item_embed, self.__uitj_placeholder[:, 2], name="t_embed_")
        # negative item embedding (N, K)
        j_embed = tf.nn.embedding_lookup(self.__item_embed, self.__uitj_placeholder[:, 3], name="j_embed_")

        # positive item bias
        i_bias = tf.nn.embedding_lookup(self.__item_bias, self.__uitj_placeholder[:, 1], name="i_bias_")
        # collaborative item bias
        t_bias = tf.nn.embedding_lookup(self.__item_bias, self.__uitj_placeholder[:, 2], name="t_bias_")
        # negative item bias
        j_bias = tf.nn.embedding_lookup(self.__item_bias, self.__uitj_placeholder[:, 3], name="j_bias_")

        ui = tf.add(tf.reduce_sum(u_embed * i_embed, reduction_indices=1), i_bias)
        ut = tf.add(tf.reduce_sum(u_embed * t_embed, reduction_indices=1), t_bias)
        uj = tf.add(tf.reduce_sum(u_embed * j_embed, reduction_indices=1), j_bias)

        utj_coef = tf.add(self.__coefs_placeholder[:,1], 1., name="utj_coef")
        uij_coef = tf.add(self.__coefs_placeholder[:,0], 1., name="uij_coef")
        uit_coef = tf.divide(uij_coef, utj_coef, name="uit_coef")

        uit = tf.reduce_sum(-tf.log(tf.sigmoid( uit_coef * (ui - ut))), name="loss_uit")
        utj = tf.reduce_sum(-tf.log(tf.sigmoid( utj_coef * (ut - uj))), name="loss_utj")
        uij = tf.reduce_sum(-tf.log(tf.sigmoid( uij_coef * (ui - uj))), name="loss_uij")

        embed_loss_ = self.__alpha * uit + self.__beta * utj + self.__gamma * uij
        return embed_loss_

    @property
    def __loss(self):
        return self.__embed_loss__  + self.__reg_loss__

    @property
    def __predict__(self):
        tst_user_embed = tf.nn.embedding_lookup(self.__user_embed, self.__users_placeholder, name="tst_user_embed_")
        predicts = tf.matmul(tst_user_embed, tf.transpose(self.__item_embed), name="predicts") + tf.expand_dims(self.__item_bias,0)
        return predicts

    @property
    def __optimize__(self):
        gds = []
        gds.append(tf.train.AdagradOptimizer(self.__lr).minimize(self.__loss, var_list=[self.__user_embed, self.__item_embed, self.__item_bias]))
        with tf.control_dependencies(gds):
            return gds + [self.__user_embed, self.__item_embed, self.__item_bias]

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

    def train(self, fold, trasR, tstsR):
        # similarity matrix
        self.__simMat = self.__topk__(self.__calsim__(trasR))
        self.__coefMat = self.__calcoef__(trasR)
        # print(self.__coefMat)

        ave = self.__coefMat.sum() / self.__coefMat.nnz
        print('fold=%d' % fold, '__coefMat:', 'shape=', self.__coefMat.shape, 'nnz=%d' % self.__coefMat.nnz,
              'sum=%.6f' % self.__coefMat.sum(),
              'ave=%.6f' % ave,
              'mean=%.6f' % self.__coefMat.mean())

        # global
        # self.__coefMat /= ave
        # or user
        for i in range(self.__coefMat.shape[0]):
            ave = self.__coefMat[i,:].sum()/self.__coefMat[i,:].nnz
            if ave>0:
                self.__coefMat[i, :] /= ave

        # self.__coefMat = self.__coefMat.multiply(self.__coefMat>.1)
        # self.__coefMat = self.__coefMat + trasR

        # # # stat for plot
        # tstsR_ = lil_matrix(tstsR)
        # __coefMat_ = lil_matrix(self.__coefMat)
        # heat_X=10
        # stat=[0 for x in range(heat_X)]
        # coefMat_rows = __coefMat_.rows
        # tst_rows = tstsR_.rows
        # m, n = 0, 0
        # for u, row in enumerate(coefMat_rows):
        #     if len(row)<heat_X:
        #         continue
        #     if len(set(tst_rows[u]) & set(row))==0:
        #         continue
        #     vals = [-self.__coefMat[u,i] for i in row] # row
        #     ids = np.argsort(vals)
        #     len_ = int(len(ids)/heat_X)
        #     m += 1.
        #     n += len(row)
        #     for id in ids:
        #         if row[id] in set(tst_rows[u]):
        #             stat[min(heat_X-1,int(id/len_))] += 1.
        # print(m, round(n/m,2), [round(val/(m),2) for val in stat])
        # exit()





        self.__sampler = Sampler(trasR, self.__coefMat, self.__batch_size)

        gc.collect()

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
                starttime = dt.datetime.now()
                losses = []
                for _ in range(n_batches):
                    uitj_batch, coef_batch = self.__sampler.next_batch()
                    _, loss = self.__sess.run(train_op, {self.__uitj_placeholder: uitj_batch,
                                                   self.__coefs_placeholder: coef_batch})
                    losses.append(loss)
                endtime = dt.datetime.now()
                aveloss = np.mean(losses)

                yss_pred = self.__recommend(test_users, tstintra_set)
                scores = self.__eval(yss_true, yss_pred)
                print(dt.datetime.now().strftime('%m-%d %H:%M:%S'),
                        "%s_fold=%d iter=%2d:" % (self.__split_method, fold, iter + 1),
                      "TraLoss=%.4f lr=%.4f" % (aveloss, self.__lr),
                      '\tTst@' + str(self.__topN) + ':' + ' '.join(
                          [eval_metric + '=%.4f' % (score) for eval_metric, score in zip(self.__eval_metrics, scores)]),
                "\t\ttimecost=%d(s)" %  (endtime-starttime).seconds)

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
