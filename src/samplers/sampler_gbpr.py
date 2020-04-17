
import numpy as np
from threading import Thread
from multiprocessing import Queue

# user: posItem: n negItems
class Sampler(object):
    def __init__(self, trasR, gsize=2, n_neg=5, batch_size=100, n_workers=1):
        self.batch_size, self.gsize, self.n_neg = batch_size, gsize, n_neg

        self.n_users, self.n_items = trasR.shape[0], trasR.shape[1]

        self.useritem_pairs = np.array(trasR.nonzero()).T
        self.user_posItemset = {u: set(row) for u, row in enumerate(trasR.rows)}
        self.item_posUserList = {i: list(col) for i, col in enumerate(trasR.transpose().rows)}

        self.result_queue = Queue(maxsize=n_workers*2)
        self.threads = []
        for i in range(n_workers):
            self.threads.append(Thread(target=self.__sample_function__))
            self.threads[-1].start()

    def __sample_function__(self,):
        while True:
            np.random.shuffle(self.useritem_pairs)
            for i in range(int(len(self.useritem_pairs) / self.batch_size)):
                # positive item samples
                useritem_pairs_batch = self.useritem_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]

                # negative item samples
                negItems_batch = np.random.randint(0, self.n_items, size=(len(useritem_pairs_batch), self.n_neg))

                # group users
                group_batch = np.random.randint(0, self.n_users, size=(len(useritem_pairs_batch), self.gsize))

                for ind, user_posItem, negItems_, group in zip(range(len(useritem_pairs_batch)), useritem_pairs_batch, negItems_batch, group_batch):
                    u, i = user_posItem
                    for ind_, negItem in enumerate(negItems_):
                        while negItem in self.user_posItemset[u]:
                            negItems_batch[ind, ind_] = negItem = np.random.randint(0, self.n_items)
                    group_batch[ind] = group = np.random.choice(self.item_posUserList[i], self.gsize)

                self.result_queue.put((useritem_pairs_batch, negItems_batch, group_batch))

    def next_batch(self):
        return self.result_queue.get()
