# Project for Master Thesis: 
### Engagement Measures: Investigating the Influence on User Subscription Behavior in Digital News Media based on Clickstream Data
This is a project, which I developed in the context of my master thesis. 
The different ML models, analyses and results serve as a basis for the written work.
The steps from the raw data set to the final results are divided into two transformation groups. 
For the efficient processing of Big Data, important cleaning and aggregation transformations are 
performed in a `spark` environment with a Big Data tool. After `spark` is used to create a dataset suitable for local processing, 
further steps are performed in a `pandas` environment. The reason for this is that many ML packages are not 
compatible with `spark` and `pandas` provides more flexible options.

**Edit: For the public repository the data is removed. If you are interested in the data, please contact 
the author: lukas.hub(at)gmx.net. When the data is available, it can be moved to the designated directories to run the code.**

The raw dataset used for the work is clickstream data from a news media platform. Since the dataset is 
several terabytes in size, there is only one sample avaiable for the repository, which contains about 4.5 million 
events. It needs to be placed in:
```sh
master_thesis/engagement/preparation/data
```

The training and test data set used for the transformations in `pandas` can be provided in their original size. These must be placed in the following directory:
```sh
master_thesis/engagement/machine_learning/data
```

The different ML models that are trained are stored under 
```sh
master_thesis/engagement/machine_learning/models
```
so that results are reproducible.  Furthermore, visualizations, which are also used in the work, are stored under
```sh
master_thesis/engagement/machine_learning/figures 
```
To ensure successful execution of the python files, Python version 3.7.3 must be used. At the moment, one of 
the packages used has problems with the latest Python version 3.9.4.

When executing the Python files, the following is the correct order:
```sh
# install needed packages
$ pip install -r master_thesis/requirements.txt
# aggregation from events on session level
$ master_thesis/engagement/preparation/session_preparation.py
# create features based on browsing behaviour
$ master_thesis/engagement/preparation/feature_engineering.py
# final filter
$ master_thesis/engagement/preparation/final_for_pandas.py
# preprocessing for ml models
$ master_thesis/engagement/machine_learning/preprocessing.py
# perform correlation analysis
$ master_thesis/engagement/machine_learning/correlation_analysis.py
# train or load models
$ master_thesis/engagement/machine_learning/(one_of_the_models).py
$ master_thesis/engagement/machine_learning/(one_of_the_models).py
$ ...
```