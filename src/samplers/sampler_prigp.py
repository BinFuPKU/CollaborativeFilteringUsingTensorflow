
import numpy as np
from threading import Thread
from multiprocessing import Queue

# user: posItem: n negItems
class Sampler(object):
    def __init__(self, trasR, coefMat, batch_size=100, n_workers=1):
        self.batch_size, self.coefMat = batch_size, coefMat

        self.n_users, self.n_items = trasR.shape[0], trasR.shape[1]

        self.useritem_pairs = np.array(trasR.nonzero()).T
        self.user_posItemset = {u: set(row) for u, row in enumerate(trasR.rows)}
        self.user_coefItemset = {u: set(row) for u, row in enumerate(coefMat.rows)}
        self.user_coefItemset_vals = {u: set([coefMat[u,ind] for ind in row]) for u,row in enumerate(coefMat.rows)}

        self.result_queue = Queue(maxsize=n_workers*2)
        self.threads = []
        for i in range(n_workers):
            self.threads.append(Thread(target=self.__sample_function__))
            self.threads[-1].start()

    def __sample_function__(self,):
        while True:
            np.random.shuffle(self.useritem_pairs)
            for i in range(int(len(self.useritem_pairs) / self.batch_size)):
                batch = np.zeros((self.batch_size, 5), dtype=int)
                # positive item samples
                batch[:,:2] = self.useritem_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]
                # negative item samples
                batch[:, 2] = np.random.randint(0, self.n_items, size=self.batch_size)
                for ind in range(self.batch_size):
                    u, i, j = batch[ind,0], batch[ind,1], batch[ind,2]
                    while j in self.user_posItemset[u]:
                        batch[ind,2] = j = np.random.randint(0, self.n_items)

                    t, k = i, j
                    if len(self.user_coefItemset_vals[u])>0:
                        t = np.random.choice(list(self.user_coefItemset[u]),1)[0]
                        k = np.random.randint(self.n_items) # default 0 group
                        while k in self.user_coefItemset[u]:
                            k = np.random.randint(self.n_items)

                        if len(self.user_coefItemset_vals[u])>1 and np.random.randn()<self.coefMat[u, :].nnz/float(self.n_items):
                            k = np.random.choice(list(self.user_coefItemset[u]),1)[0]
                            while self.coefMat[u, t] == self.coefMat[u, k]:
                                k = np.random.choice(list(self.user_coefItemset[u]), 1)[0]
                            if self.coefMat[u, t] < self.coefMat[u, k]:
                                t, k = k, t
                    batch[ind, :] = [u,i,j,t,k]
                self.result_queue.put(batch)

    def next_batch(self):
        return self.result_queue.get()
