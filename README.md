# Collaborative Filtering Using Tensorflow
Realising several collaborative filtering methods in recommendation domain (done in 2019).
The code is relatively readable, and you can download and run it without any modification. It is divided into parts of evaluation, parallel sampling using multi-process, model and test, with considering using sparse matrix and CPU / GPU mode switching. If you feel helpful for your work, please star this project or watch my future work!

**Running Requirements:**
* python 3.6+
* tensorflow 1.13+


We implement some common-used collaborative filtering methods in recommendation domain, including:

* [1]. Sarwar, B.M., Karypis, G., Konstan, J.A., Riedl, J.: Item-based collaborative filtering recommendation algorithms. In: WWW. pp. 285–295. ACM (2001) (code: itemcf.py)

* [2]. Pan, R., Zhou, Y., Cao, B., Liu, N.N., Lukose, R.M., Scholz, M., Yang, Q.: One- class collaborative filtering. In: ICDM. pp. 502–511. IEEE Computer Society (2008) (code: testwrmf.py)

* [3]. Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L.: BPR: bayesian personalized ranking from implicit feedback. In: UAI. pp. 452–461. AUAI Press (2009) (code: bprmf.py)

* [4]. Pan, W., Chen, L.: GBPR: group preference based bayesian personalized ranking for one-class collaborative filtering. In: IJCAI. pp. 2691–2697. IJCAI/AAAI (2013) (code: gbprmf.py)

* [5]. Qiu,S.,Cheng,J.,Yuan,T.,Leng,C.,Lu,H.:Item group based pairwise preference learning for personalized ranking. In: SIGIR. pp. 1219–1222. ACM (2014) (code: prigp.py)

* [6]. Hsieh,C.,Yang,L.,Cui,Y.,Lin,T.,Belongie,S.J.,Estrin,D.:Collaborative metric learning. In: WWW. pp. 193–201. ACM (2017) (code: cml.py)

* [7]. Liu, H., Wu, Z., Zhang, X.: CPLR: collaborative pairwise learning to rank for personalized recommendation. Knowl.-Based Syst. 148, 31–40 (2018) (code: cplr_u.py)

* [8]. Yi Tay, Luu Anh Tuan, Siu Cheung Hui: Latent Relational Metric Learning via Memory-based Attention for Collaborative Ranking. WWW 2018: 729-739. (code: lrml.py)


**The project architecture:**

----CollaborativeFilteringUsingTensorflow

    |----data
    |    |----movielens
    |    |    |----ml-100k (toy dataset, five folds for cross-validation)
    |    |    |    |----ratings.txt
    |    |    |    |----ratings_.txt
    |    |    |    |----ratings__1_tra.txt
    |    |    |    |----ratings__1_tst.txt
    |    |    |    |----ratings__2_tra.txt
    |    |    |    |----ratings__2_tst.txt
    |    |    |    |----ratings__3_tra.txt
    |    |    |    |----ratings__3_tst.txt
    |    |    |    |----ratings__4_tra.txt
    |    |    |    |----ratings__4_tst.txt
    |    |    |    |----ratings__5_tra.txt
    |    |    |    |----ratings__5_tst.txt
    |----src
    |    |----code_data.py (code the user id and item id into 0,1,2... for indexing in matrix)
    |    |----cut_data.py (split the dataset into k folds (e.g. 5) for cross-validation, each one includes training set and test set)
    |    |----metrics (evaluation metrics)
    |    |    |----rating.py (evaluation metrics for rating, e.g. MAE/RMSE/MSE)
    |    |    |----ranking.py  (evaluation metrics for ranking, e.g. MAP/MRR/NDCG/AUC/HR/ARHR)
    |    |----utils (some general userful functions used as tools)
    |    |    |----Util.py 
    |    |    |----IOUtil.py (read and write files)
    |    |----models (recommendation models and their corresponding test files, CPU/GPU mode switching)
    |    |    |----PL (preference learning models)
    |    |    |    |----testcplr_u.py
    |    |    |    |----models
    |    |    |    |    |----bprmf.py
    |    |    |    |    |----gbprmf.py
    |    |    |    |    |----prigp.py
    |    |    |    |    |----cml.py
    |    |    |    |    |----cplr_u.py
    |    |    |    |----testbprmf.py
    |    |    |    |----testgbprmf.py
    |    |    |    |----testprigp.py
    |    |    |    |----testcml.py
    |    |    |    |----testcplr_u.py
    |    |    |----basic (basic recommendation models)
    |    |    |    |----models
    |    |    |    |    |----pop.py
    |    |    |    |    |----usercf.py
    |    |    |    |    |----itemcf.py
    |    |    |    |    |----mf.py
    |    |    |    |    |----svd.py
    |    |    |    |    |----wrmf.py
    |    |    |    |----testpop.py
    |    |    |    |----testucf.py
    |    |    |    |----testicf.py
    |    |    |    |----testmf.py
    |    |    |    |----testsvd.py
    |    |    |    |----testwrmf.py
    |    |    |----others
    |    |    |    |----models
    |    |    |    |    |----useritemcf.py
    |    |    |    |    |----amf.py
    |    |    |    |    |----lrml.py
    |    |    |    |----testuicf.py
    |    |    |    |----testamf.py
    |    |    |    |----testlrml.py
    |    |----samplers (different samplers for models, parallel sampling using multi-process)
    |    |    |----sampler_rating.py
    |    |    |----sampler_ranking.py
    |    |    |----sampler_gbpr.py
    |    |    |----sampler_prigp.py
    |    |    |----sampler_rating_weight.py
    |    |    |----sampler_uiktj_ranking.py
    |    |    |----sampler_uitj_ranking.py
    |    |    |----sampler_uitj_ranking_.py
    |    |    |----sampler_lrml_ranking.py
    |    |    |----sampler_uij_ranking.py


**Running order:**
* code_data.py
* cut_data.py
* testbprmf.py (the test file of your model)
