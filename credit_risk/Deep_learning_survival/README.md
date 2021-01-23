# Deep Learning for Survival Analysis in Credit Risk Modelling: A Benchmark Study
> This repository contains the python implementation for the expirment part in the master thesis, and also the datasets the have been used, for replication purposes. 

## Table of contents
* [Abstract](#abstract)
* [Methodes](#methods)
* [Requirements](#requirements)
* [Data](#data)
* [References](#references)

## Abstract
Survival analysis is a hotspot in statistical research for modelling time-to-event information
with data censorship handling, which has been widely used in many applications such as
clinical research, information system and other fields with survivorship bias. Many works
have been proposed for survival analysis ranging from traditional statistic methods to machine
learning and deep learning methods. This paper examines novel deep learning techniques for
survival analysis in credit risk modelling context. After surveying through literature for deep
learning survival analysis models in various domains and categorizing them, we evaluate the
adequacy of six different models representing different categories, using two datasets of US
mortgages from separate sources. The performance of these models is evaluated using the
discrimination metric, concordance index.

## Methodes 
The repository contains python implementation of six different models, namely:
* Deepsurv (Katzman et al., 2018)<a href="#references">[1]</a>, was implemented using [pycox](https://github.com/havakv/pycox) package.
* Cox-Time (Kvamme et al., 2019)<a href="#references">[2]</a>, was implemented using [pycox](https://github.com/havakv/pycox) package.
* DeepHit (Lee et al., 2018)<a href="#references">[3]</a>, the orignal code published by the author on their respective [GitHub](https://github.com/chl8856/DeepHit) was used and adjustments were madeto the code in order to adopt it for our experiment.
* Nnet-survival (Gensheimer & Narasimhan, 2019)<a href="#references">[4]</a>, was implemented using [pycox](https://github.com/havakv/pycox) package.
* DRSA (Ren et al., 2019a)<a href="#references">[5]</a>, the orignal code published by the author on their respective [GitHub](https://github.com/rk2900/DRSA) was used and adjustments were made to the code in order to adopt it for our experiment.
* DATE (Chapfuwa et al., 2018)<a href="#references">[6]</a>, the orignal code published by the author on their respective [GitHub](https://github.com/paidamoyo/adversarial_time_to_event) was used and adjustments were made to the code in order to adopt it for our experiment.


## Requirements
The following package versions have been used to develop this work.
```
python==3.7.9
lifelines==0.25.4
pandas==1.1.4
pycox==0.2.1
scikit-learn==0.24.1
torch==1.7.0
matplotlib==3.3.3

DATE and Deephit:
tensorflow==1.15.0

DRSA: 
tensorflow==2.0.0
```
## Data
We consider the following datasets:
- [M1 (U.S mortgage data provided by International Financial Research)](http://www.internationalfinancialresearch.org)
- M2 (single-family US mortgage loans collected by (Blumenstock et al., 2020) <a href="#references">[7]</a>).

The [*datasets*](./datasets) directory contains M1 dataset (named as mortgage) and ten batches from M2 (named as data batches).

In the case of DRSA model, the model requires that the input data to be in the form of multi-hot encoded feature vector including a series of one-hot
encoded features, so the preprocessed data for DSRA can be found in [*data*](./DRSA/data) directory inside DRSA directory. For more information on the data preprocessing process, the python notebooks inside [*preprocessing*](./DRSA/preprocessing) directory are provided.

## Implementation

For each model a directory have been created that contains the python notebooks: 
* Deepsurv: contains two notebooks:
  * M1_Deepsurv: is the model implementation for M1 dataset, a change of the path to file location in the script is required, also the event of interest (default or payoff) should be modified in order to get the aimed results.
  * M2_Deepsurv: is the model implementation for M2 dataset, a change of the path to file location in the script is required and chosing the batch, also the event of interest (default or payoff) should be modified in order to get the required results.
* Cox-Time and Nnet-survival have the same logic as Deepsurv.
* Deephit: contains one python notebook, in order to replicate the results, a change of the path to file location in the script is required. After that the varibale data_mode should be modified in order to decide on the dataset and the event of interest (default or payoff), 
 * passing mort_d to the data_mode, M1 dataset is chossen with default as the event of interest,
 * passing mort_p to the data_mode, M1 dataset is chossen with payoff as the event of interest,
 * passing ndb_d to the data_mode, M2 dataset is chossen with default as the event of interest, and in this case an additional variable have to be set, variable data_number, which set the number of the batch of interest of M2 dataset, which are ten batches.  
 * passing ndb_p to the data_mode, M2 dataset is chossen with payoff as the event of interest, and again an additional variable have to be set, variable data_number, which set the number of the batch of interest of M2 dataset, which are ten batches.

## References
  \[1\] Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., & Kluger, Y. (2018). Deepsurv:
Personalized treatment recommender system using a cox proportional hazards deep
neural network. BMC Medical Research Methodology, 18(1), 24. 

  \[2\] Kvamme, H., Borgan, Ø., & Scheel, I. (2019). Time-to-event prediction with neural networks
and cox regression. Journal of Machine Learning Research, 20(129), pp. 1–30. 

  \[3\] Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). Deephit: A deep learning approach
to survival analysis with competing risks. In Thirty-second AAAI Conference on Artificial
Intelligence (AAAI-18). 

  \[4\] Gensheimer, M. F., & Narasimhan, B. (2019). A scalable discrete-time survival model for neural
networks. PeerJ, 7:e6257. 

  \[5\] Ren, K., Qin, J., Zheng, L., Yang, Z., Zhang,W., Qiu, L., & Yu, Y. (2019a). Deep recurrent survival
analysis. In Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19), 33(01), pp.
4798–4805. 

  \[6\] Chapfuwa, P., Tao, C., Li, C., Page, C., Goldstein, B., Carin, L., & Henao, R. (2018). Adversarial
time-to-event modeling. In Proceedings of the 35 the International Conference on Machine
Learning, 80, pp. 735–744. 

  \[7\] Blumenstock, G., Lessmann, S., & Seow, H.-V. (2020). Deep learning for survival and competing
risk modelling. Journal of the Operational Research Society. https://doi.org/10.1080/01605682.2020.1838960

