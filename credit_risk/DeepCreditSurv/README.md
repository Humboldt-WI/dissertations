# DeepCreditSurv: Benchmarking Deep Survival Analysis Models in Credit Risk Modelling
> A repository for the benchmarking of Deep Survival Models in Credit Risk Context.

## Models Implemented:
* Deepsurv (Katzman et al., 2018) <a href="#references">[1]</a>
* Cox-Time (Kvamme et al., 2019) <a href="#references">[2]</a>
* DeepHit (Lee et al., 2018) <a href="#references">[3]</a>
* Nnet-survival (Gensheimer & Narasimhan, 2019) <a href="#references">[4]</a>
* DRSA (Ren et al., 2019) <a href="#references">[5]</a>
* DATE (Chapfuwa et al., 2018) <a href="#references">[6]</a>
* Deep Survival Machines (Nagpal et al., 2021) <a href="#references">[7]</a>
* Deep Hazard (Rava et al., 2020) <a href="#references">[8]</a>
* Gradient Boosting Survival Tree (Bai et al., 2021) <a href="#references">[9]</a>

## Data Sets
We collected four datasets in credit risk context in our project:
* [M1 (U.S mortgage data provided by International Financial Research)](http://www.internationalfinancialresearch.org)
* M2 (single-family US mortgage loans collected by (Blumenstock et al., 2020)<a href="#references">[10]</a>)
* [Paipaidai(拍拍贷) Data provided by Heywhale](https://www.heywhale.com/mw/dataset/58c614aab84b2c48165a262d)
* [The Lending Club Data](https://www.kaggle.com/sonujha090/xyzcorp-lendingdata?select=XYZCorp_LendingData.txt)

## Experiment
We create a class for each of the models we collect and to get the test result simple create a model with dataset and file path parameters. Then execute build_model() function, the experiment will run automatically.
Example Usage:
``` python
import os
import pandas as pd
path = "D:\DeepCreditSurv"
os.chdir(path)

import DeepCreditSurv.models.DRSA.DRSA as drsa
drsa_m1 = drsa.DRSA(dataset='M1',file_path="D:\DeepCreditSurv\DeepCreditSurv\datasets\M1\WideFormatMortgageAfterRemovingNull.csv")
drsa_m1.build_model()
```

## References
  [1] Katzman, J. L., Shaham, U., Cloninger, A., Bates, J., Jiang, T., & Kluger, Y. (2018). Deepsurv:
Personalized treatment recommender system using a cox proportional hazards deep
neural network. BMC Medical Research Methodology, 18(1), 24. 

  [2] Kvamme, H., Borgan, Ø., & Scheel, I. (2019). Time-to-event prediction with neural networks
and cox regression. Journal of Machine Learning Research, 20(129), pp. 1–30. 

  [3] Lee, C., Zame, W. R., Yoon, J., & van der Schaar, M. (2018). Deephit: A deep learning approach
to survival analysis with competing risks. In Thirty-second AAAI Conference on Artificial
Intelligence (AAAI-18). 

  [4] Gensheimer, M. F., & Narasimhan, B. (2019). A scalable discrete-time survival model for neural
networks. PeerJ, 7:e6257. 

  [5] Ren, K., Qin, J., Zheng, L., Yang, Z., Zhang,W., Qiu, L., & Yu, Y. (2019a). Deep recurrent survival
analysis. In Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19), 33(01), pp.
4798–4805. 

  [6] Chapfuwa, P., Tao, C., Li, C., Page, C., Goldstein, B., Carin, L., & Henao, R. (2018). Adversarial
time-to-event modeling. In Proceedings of the 35 the International Conference on Machine
Learning, 80, pp. 735–744. 

  [7] Nagpal, C., X. Li, and A. Dubrawski (2021a).  “Deep Survival Machines: Fully Parametric
Survival Regression and Representation Learning for Censored Data with Competing Risks,” IEEE Journal of Biomedical and Health Informatics.

  [8] Rava, D., & Bradic, J. (2020). DeepHazard: neural network for time-varying risks. arXiv preprint arXiv:2007.13218.

  [9] Bai, M., Zheng, Y., & Shen, Y. (2021). Gradient boosting survival tree with applications in credit scoring. Journal of the Operational Research Society, 1-17.

  [10] Blumenstock, G., Lessmann, S., & Seow, H.-V. (2020). Deep learning for survival and competing
risk modelling. Journal of the Operational Research Society. https://doi.org/10.1080/01605682.2020.1838960
