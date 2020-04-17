
import numpy as np
from threading import Thread
from queue import Queue

# user: posItem: n negItems
class Sampler(object):
    def __init__(self, trasR, coefMat_k, coefMat_t, batch_size=1000, n_workers=1):
        self.batch_size, self.coefMat_k, self.coefMat_t = batch_size, coefMat_k, coefMat_t

        self.n_users, self.n_items = trasR.shape[0], trasR.shape[1]

        self.ui = [set(row) for row in trasR.rows]

        self.uk = [set(coefMat_k[ind,:].nonzero()[1])-self.ui[ind] for ind in range(self.n_users)]
        self.ut = [set(coefMat_t[ind,:].nonzero()[1])-self.ui[ind] for ind in range(self.n_users)]


        self.result_queue = Queue(maxsize=n_workers*2)
        self.threads = []
        for i in range(n_workers):
            self.threads.append(Thread(target=self.__sample_function__))
            self.threads[-1].start()

    def __sample_function__(self,):
        while True:
            for batch in range(int(self.n_users / self.batch_size)):
                uiktj_batch, coef_batch = [], []
                for ind in range(self.batch_size):
                    u = np.random.randint(0, self.n_users, 1)[0]
                    while len(self.ui[u])==0 or len(self.uk[u])==0 or len(self.ut[u])==0 or \
                            len(self.ui[u])+len(self.uk[u])+len(self.ut[u]) >= self.n_items:
                        u = np.random.randint(0, self.n_users, 1)[0]
                    i = np.random.choice(list(self.ui[u]),1)[0]
                    k = np.random.choice(list(self.uk[u]),1)[0]
                    t = np.random.choice(list(self.ut[u]),1)[0]

                    j = np.random.randint(0, self.n_items, 1)[0]
                    while j in self.ui[u] or j in self.uk[u] or j in self.ut[u]:
                        j = np.random.randint(0, self.n_items, 1)[0]
                    uiktj_batch.append(np.array([u,i,k,t,j]))
                    coef_batch.append(np.array([self.coefMat_k[u,i],self.coefMat_k[u,k],self.coefMat_t[u,t]]))
                uiktj_batch, coef_batch = np.asarray(uiktj_batch), np.asarray(coef_batch)
                self.result_queue.put((uiktj_batch, coef_batch))

    def next_batch(self):
        return self.result_queue.get()
