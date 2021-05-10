# Addressing the Long-Tail Products in Customer Purchase Prediction

Master Thesis submitted to the School of Business and Economics of Humboldt-Universität zu Berlin for the degree M.Sc. Economics and Management Science. 


## Abstract
Recommendation systems are valuable in marketing research as they provide information to retailers to predict what a customer will buy next. A common problem in recommendation systems is it often suffers from a lack of novelty and diversity in recommendations. The long-tail products refer to less popular or newly added products in the assortment (Alshammari et al., 2017). Long-tail recommendations can improve personalized recommendations and help the retailer avoid perishable products to go waste by recommending to potential customers. Addressing long-tail products in recommendation systems is often overlooked in offline grocery stores due to shelf and storage capacity. Yet, recent research shows that neglecting the long-tail of products in offline grocery stores is an outdated approach since the customer might be drawn away in case of a limited number of choices (Hoskins, 2020). This thesis focuses on recommendation systems for a wide selection of products obtained from scanned data from a grocery retailer. The proposed architecture combines Multi-layer Perceptron for users, products, date, and price attributes and Convolutional Neural Network for modeling purchase histories. The model showed higher weighted AUC and weighted Recall scores than the benchmark models. Moreover, the predictive power of the recommendation model seems to increase when applied to more frequent visitors with better and diverse purchase histories.

----------------------------------------
The data source used in this thesis is not to be shared due to privacy concerns. Below, we explain how one can utilize the code provided and what types of data would be necessary to proceed.

- Data_Import_and_Cleaning.ipynb : 
Requires two data sources: 
    * Transactions data containing user id, product id, basket id, price and day of purchase.
    * Product attributes data containing product id, product description, category description. 
It outputs data_cleaned.pkl for filtered and cleaned transactions and word_vectors_saved.pkl from weights trained on word2vec.
- Negative_Sampling_and_Feature_Engineering.ipynb: This notebooks uses data_cleaned.pkl from the previous notebook to preprocess the data further for modeling and product attributes data. It outputs data_processed.pkl.
- Modeling_MF_Shopper_WDL_LSTM.ipynb: Requests data_processed.pkl, product attributes data and word_vectors_saved.pkl.
- Modeling_CNN_elasticity.ipynb: Requests data_processed.pkl, product attributes data and word_vectors_saved.pkl

Please note the notebooks are adjusted to read data from Google Drive and to run on Google Colab. To use the notebooks locally, the directories must be adapted accordingly.