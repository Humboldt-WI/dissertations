# TS_Transformer_Finance - Master Thesis (Business Administration, M.Sc.)
Experiment code and source raw data as well as selected experiment result data for analysis.

## Transformer Architecture in Deep Learning for Financial Price Forecasting
### Abstract
Time series prediction plays an important part in many fields of scientific research and professional applications, including the financial sector. In this paper we examine the recently introduced transformer network architecture for natural language processing and apply a time series adapted transformer model to a financial use case. While providing a brief overview of commonly used methods for predictive task on time series data, the paper’s focus lies on the adaption of the new transformer architecture to this purpose and a subsequent analysis of its performance. For this, we construct an experiment where the transformer is trained on stock market time series and its subsequent prediction output is used to decide on actions in a downstream trading model. For reference, we compare the transformer’s performance to an LSTM’s performance on the same data and descent into an ablation study on the transformer’s adaptations.

Keywords:  *Transformer, LSTM, Finance, Time Series Prediction*

The accomanying thesis paper is available from: **XXXXXXXX**

### Experiment Description
The objective of the thesis paper was to implement a transformer for financial price prediction. the transformer architecture was first proposed in the paper *Attention is all you need* by Vaswani et al. (2017) and origninally developed for natural language processing (NLP). This repository contains the experimental code including the implementation of the transformer architecture, that was constructed based on the orininal NLP architecture with selected adaptations to the aim of financial price prediction. 
To evaluate model performance the experimental framework converts raw financial predictions into trading choices for a simplified trading strategy.
In the experiment, the transformer model was first assessed on its performance on this task. Secondly, transformer- and, as a reference point, LSTM-based trading performance were contrasted and the experiment conculded  a short ablation study on the transformer components that were added to ready the original NLP model for time-series forecasting.

### Code and Data Structure
The experiment was written in python code which is provided in the repository for download and execution in a standard python environment (Compatibility tested for Python 3.8). Necessary packages are listed in the **requirements doc**. 
The experiment itself was partitioned into the following: 

**- Transformer Part** [Transformer_main](Transformer_main.py): includes data preprocessing, predictive transformer model and trading model

**- LSTM Part** [LSTM_main](LSTM_main.py): includes data preprocessing, predictive LSTM model and trading model

**- Monkey Part** [Monkey_main](Monkey_main.py): includes supplementary random guessing 'monkey' as predictive model and trading model

**- Analyis Part** [Analysis_main](Analysis_main.py): includes code for descriptive analysis of experiment results

**- Graphics Part** [Graphics_main](Graphics_main.py): includes code to plot selected results

In addition supplementary code, raw input files as well as exemplary results from the thesis paper are included in [resources](resources/) and [experiment](experiment/).

#### Execute Code
To run the code, I recommend using a IDE as most parts are generalized and require parameter selection at the top of the main script. Otherwise, simply save changes in the main file before executing the file.

The [transformer](Transformer_main.py), [LSTM](LSTM_main.py) and [Monkey](Monkey_main.py) part all run prediction and trading experiments on the dataset, that in retunrn create and save experiment output files to the [experiment](experiment/) directory. To perform a statistical analysis on these experiment results run [Analysis](Analysis_main.py). Outputs from the analysis can subsequentially plottet with [Graphics](Graphics_main.py). 
Make sure to recreate the same directory structure when downloading the data as all main scripts need to import relevant resources and data from their relative data paths.

#### Modify Parameters
Every main script includes a in-code menu for selecting the exact experiment and or setting model parameters.
```python
EID = 'Test'                    # Identifier for experiment run. Used to identify results for analysis.
""" model parameters """
window_size = 20                # Window size for lookback in sequences. (def. 80; arbitrary int)
V = 200                         # Value, Query, Key dimension ~ vocab size. (def: 200; int)
N = 6                           # Encoder and decoder layers in stack. (def: 6; int)
dropout = 0.1                   # Dropout probability. (def: 0.1; float) 
convolve = 3                    # Convolution kernel in MHA-module. (def: 3; int; 0 ~ no convolution layer)
...cont.
```
Paramter descriptions and available or standard values used in the paper (def.) are provided in the respective scripts. A full list is available in [Parameter_Settings](Parameter_settings.csv).
The *EID* - is the identifier for an experiment run. Choose a unique EID per experiment run to not overwrite previous results. To load experiment results in the subsequent analysis or plotting modules insert the respective EID (or EIDs).

#### Experiment Results
Detailed findings and interpretaions from the experiment are included in the thesis paper. The raw experiment data is equally provided in the experiment directory and can be used as input to the analysis and plotting modules.

![plot 2](/TS_Transformer_Finance/analysis/1_monkey_SP500_distribution_p_return.png)
Transformer-based trading (green) showed significantl better returns than simple market long (orange, incl. 1 and 99 percentiles) or random long-short positions (blue, incl. 1 and 99 percentiles). 

![plot 1](/TS_Transformer_Finance/analysis/1_9_11_cum_profit.png)
However, overall the transformer model was not ale to outperform the LSTM-reference.

