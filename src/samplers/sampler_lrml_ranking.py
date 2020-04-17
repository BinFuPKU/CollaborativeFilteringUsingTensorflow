
import numpy as np
from threading import Thread
from multiprocessing import Queue

# user: posItem: n negItems
class Sampler(object):
    def __init__(self, trasR, batch_size=1000, n_workers=1):
        self.trasR, self.batch_size = trasR, batch_size

        self.n_users, self.n_items = trasR.shape[0], trasR.shape[1]
        self.useritem_pairs = np.array(trasR.nonzero()).T

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
                pos_useritem_batch = self.useritem_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]

                # negative item samples
                neg_useritem_batch = []
                for ind in range(self.batch_size):
                    u,i = np.random.randint(0, self.n_users, 1)[0],np.random.randint(0, self.n_items, 1)[0]
                    while self.trasR[u,i]>0:
                        u,i = np.random.randint(0, self.n_users, 1)[0],np.random.randint(0, self.n_items, 1)[0]
                    neg_useritem_batch.append(np.array([u,i]))
                pos_useritem_batch, neg_useritem_batch = np.asarray(pos_useritem_batch), np.asarray(neg_useritem_batch)
                self.result_queue.put((pos_useritem_batch, neg_useritem_batch))

    def next_batch(self):
        return self.result_queue.get()
