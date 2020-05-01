# CollaborativeFilteringUsingTensorflow
Realising several collaborative filtering methods in recommendation domain (done in 2019).
The code is relatively readable. It is divided into parts of evaluation, parallel sampling using multi-process, model and test, with considering using sparse matrix and CPU / GPU mode switching.

**Running Requirements:**
* python 3.6+
* tensorflow 1.13+


We realized some common-used collaborative filtering methods in recommendation domain, including:

* [1]. Sarwar, B.M., Karypis, G., Konstan, J.A., Riedl, J.: Item-based collaborative filtering recommendation algorithms. In: WWW. pp. 285–295. ACM (2001)

* [2]. Pan, R., Zhou, Y., Cao, B., Liu, N.N., Lukose, R.M., Scholz, M., Yang, Q.: One- class collaborative filtering. In: ICDM. pp. 502–511. IEEE Computer Society (2008)

* [3]. Rendle, S., Freudenthaler, C., Gantner, Z., Schmidt-Thieme, L.: BPR: bayesian personalized ranking from implicit feedback. In: UAI. pp. 452–461. AUAI Press (2009)

* [4]. Pan, W., Chen, L.: GBPR: group preference based bayesian personalized ranking for one-class collaborative filtering. In: IJCAI. pp. 2691–2697. IJCAI/AAAI (2013)

* [5]. Qiu,S.,Cheng,J.,Yuan,T.,Leng,C.,Lu,H.:Item group based pairwise preference learning for personalized ranking. In: SIGIR. pp. 1219–1222. ACM (2014)

* [6]. Hsieh,C.,Yang,L.,Cui,Y.,Lin,T.,Belongie,S.J.,Estrin,D.:Collaborative metric learning. In: WWW. pp. 193–201. ACM (2017)

* [7]. Liu, H., Wu, Z., Zhang, X.: CPLR: collaborative pairwise learning to rank for personalized recommendation. Knowl.-Based Syst. 148, 31–40 (2018)

* [8]. Yi Tay, Luu Anh Tuan, Siu Cheung Hui: Latent Relational Metric Learning via Memory-based Attention for Collaborative Ranking. WWW 2018: 729-739.


The project architecture:

----CollaborativeFilteringUsingTensorflow

    |    |----cut_data.py (split the dataset into five-fold for cross-validation, each one includes training set and test set)
    
    |    |----code_data.py (code the user id and item id into 0,1,2... for indexing in matrix)
    
    |----data
    
    |    |----movielens
    
    |    |    |----ml-100k (toy dataset, fivefold cross-validation)
    |    |    |    |----ratings.txt
    |    |    |    |----ratings__1_tra.txt
    |    |    |    |----ratings__1_tst.txt
    |    |    |    |----ratings__2_tra.txt
    |    |    |    |----ratings__3_tst.txt
    |    |    |    |----ratings__2_tst.txt
    |    |    |    |----ratings__3_tra.txt
    |    |    |    |----ratings_.txt
    |    |    |    |----ratings__4_tst.txt
    |    |    |    |----ratings__5_tra.txt
    |    |    |    |----ratings__4_tra.txt
    |    |    |    |----ratings__5_tst.txt
    |----src
    |    |----metrics (evaludate fold)
    |    |    |----rating.py (evaluate metrics for ranking, e.g. MAP/MRR/NDCG/AUC/HR/ARHR)
    |    |    |----ranking.py  (evaluate metrics for rating, e.g. MAE/RMSE/MSE)
    |    |----utils (some general userful functions)
    |    |    |----Util.py 
    |    |    |----IOUtil.py (read and write files)
    |    |----models (recommendation models and their corresponding test files, CPU/GPU mode switching)
    |    |    |----PL (preference learning models)
    |    |    |    |----testcplr_u.py
    |    |    |    |----models
    |    |    |    |    |----gbprmf.py
    |    |    |    |    |----bprmf.py
    |    |    |    |    |----prigp.py
    |    |    |    |    |----cml.py
    |    |    |    |    |----cplr_u.py
    |    |    |    |----testprigp.py
    |    |    |    |----testbprmf.py
    |    |    |    |----testgbprmf.py
    |    |    |    |----testcml.py
    |    |    |----basic (basic recommendation models)
    |    |    |    |----testicf.py
    |    |    |    |----testsvd.py
    |    |    |    |----testmf.py
    |    |    |    |----models
    |    |    |    |    |----usercf.py
    |    |    |    |    |----wrmf.py
    |    |    |    |    |----svd.py
    |    |    |    |    |----mf.py
    |    |    |    |    |----itemcf.py
    |    |    |    |    |----pop.py
    |    |    |    |----testpop.py
    |    |    |    |----testucf.py
    |    |    |    |----testwrmf.py
    |    |    |----others
    |    |    |    |----models
    |    |    |    |    |----useritemcf.py
    |    |    |    |    |----lrml.py
    |    |    |    |    |----amf.py
    |    |    |    |----testamf.py
    |    |    |    |----testlrml.py
    |    |    |    |----testuicf.py
    |    |----samplers (different samplers for models, parallel sampling using multi-process)
    |    |    |----sampler_rating_weight.py
    |    |    |----sampler_uiktj_ranking.py
    |    |    |----sampler_prigp.py
    |    |    |----sampler_uitj_ranking.py
    |    |    |----sampler_uitj_ranking_.py
    |    |    |----sampler_rating.py
    |    |    |----sampler_lrml_ranking.py
    |    |    |----sampler_uij_ranking.py
    |    |    |----sampler_ranking.py
    |    |    |----sampler_gbpr.p
