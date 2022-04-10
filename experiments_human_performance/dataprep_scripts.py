from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer

import pandas as pd 
import numpy as np



def ohe_dataset(df_train, df_test, ohe_columns, max_unique=30):

    for col in ohe_columns:

        incomplete_cols = []

        if df_train[col].nunique() > max_unique:

            top_values = df_train[col].value_counts()[:max_unique]

            df_train.loc[~(df_train[col].isin(top_values.index)), col] = df_train[col].max()+1
            df_test.loc[~(df_test[col].isin(top_values.index)), col] = df_test[col].max()+1

            incomplete_cols += [col]


    ohe = OneHotEncoder(handle_unknown='ignore').fit(df_train[ohe_columns])


    df_ohe_train = pd.DataFrame(data = ohe.transform(df_train[ohe_columns]).astype('int32').toarray())
    df_train = pd.concat([df_train[[col for col in df_train.columns if col not in list(set(ohe_columns) - set(incomplete_cols))]], df_ohe_train], axis=1)

    df_ohe_test = pd.DataFrame(data = ohe.transform(df_test[ohe_columns]).astype('int32').toarray())
    df_test = pd.concat([df_test[[col for col in df_test.columns if col not in list(set(ohe_columns) - set(incomplete_cols))]], df_ohe_test], axis=1)


    return df_train, df_test



def load_dataset_amazon_employee(path, feature_engineering=False):

    if feature_engineering:    
        return ohe_dataset(pd.read_csv(f'{path}/train.csv').rename({'ACTION':'target'}, axis=1), \
                           pd.read_csv(f'{path}/test.csv').drop('id', axis=1),
                           ['RESOURCE', 'MGR_ID','ROLE_ROLLUP_1','ROLE_ROLLUP_2', \
                            'ROLE_DEPTNAME', 'ROLE_TITLE','ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE'])


    return pd.read_csv(f'{path}/train.csv').rename({'ACTION':'target'}, axis=1), \
           pd.read_csv(f'{path}/test.csv').drop('id', axis=1)

def submit_dataset_amazon_employee(path, prediction):
    predict_df = pd.read_csv(f'{path}/test.csv')[['id']].rename({'id':'Id'}, axis=1)
    predict_df['Action'] = prediction[:,1]

    return predict_df




def load_dataset_challenges_in_representation_learning_the_black_box_learning_challenge(path, feature_engineering = False):
    if feature_engineering:
        raise "Feature engineering do not implemented"

    return pd.read_csv(f'{path}/train.csv').rename({'label':'target'}, axis=1), \
           pd.read_csv(f'{path}/test.csv')
           
def submit_dataset_challenges_in_representation_learning_the_black_box_learning_challenge(path, prediction):
    predict_df = pd.read_csv(f'{path}/sample_submission.csv')[['Id']]
    predict_df['Class'] = prediction.astype(int)

    return predict_df






def load_dataset_otto_group_product_classification_challenge(path, feature_engineering = False):
    if feature_engineering:
        raise "Feature engineering do not implemented"

    class_list = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8','Class_9']

    le = LabelEncoder()
    le.fit(class_list)

    df_train = pd.read_csv(f'{path}/train.csv').drop('id', axis=1)
    df_train['target'] = le.transform(df_train['target'])

    return df_train, \
           pd.read_csv(f'{path}/test.csv').drop('id', axis=1)

def submit_dataset_otto_group_product_classification_challenge(path, prediction):
    class_list = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8','Class_9']

    df_predict = pd.DataFrame(columns = class_list, data = prediction)

    df_predict['id'] = pd.read_csv(f'{path}/test.csv')['id']

    return df_predict






def load_dataset_predict_who_is_more_influential_in_a_social_network(path, feature_engineering=False):
    if feature_engineering:

        def feat_eng(df):

            df['follower_diff'] = (df['A_follower_count'] > df['B_follower_count']).astype('int')
            df['following_diff'] = (df['A_following_count'] > df['B_following_count']).astype('int')
            df['listed_diff'] = (df['A_listed_count'] > df['B_listed_count']).astype('int')
            df['ment_rec_diff'] = (df['A_mentions_received'] > df['B_mentions_received']).astype('int')
            df['rt_rec_diff'] = (df['A_retweets_received'] > df['B_retweets_received']).astype('int')
            df['ment_sent_diff'] = (df['A_mentions_sent'] > df['B_mentions_sent']).astype('int')
            df['rt_sent_diff'] = (df['A_retweets_sent'] > df['B_retweets_sent']).astype('int')
            df['posts_diff'] = (df['A_posts'] > df['B_posts']).astype('int')

            df['A_pop_ratio'] = (df['A_mentions_sent']/np.maximum(1,df['A_listed_count']))
            df['A_foll_ratio'] = (df['A_follower_count']/np.maximum(1,df['A_following_count']))
            df['A_ment_ratio'] = (df['A_mentions_sent']/np.maximum(1,df['A_mentions_received']))
            df['A_rt_ratio'] = (df['A_retweets_sent']/np.maximum(1,df['A_retweets_received']))
            
            df['B_pop_ratio'] = (df['B_mentions_sent']/np.maximum(1,df['B_listed_count']))
            df['B_foll_ratio'] = (df['B_follower_count']/np.maximum(1,df['B_following_count']))
            df['B_ment_ratio'] = (df['B_mentions_sent']/np.maximum(1,df['B_mentions_received']))
            df['B_rt_ratio'] = (df['B_retweets_sent']/np.maximum(1,df['B_retweets_received']))
            
            df['A/B_foll_ratio'] = (df['A_foll_ratio'] > df['B_foll_ratio']).astype('int')
            df['A/B_ment_ratio'] = (df['A_ment_ratio'] > df['B_ment_ratio']).astype('int')
            df['A/B_rt_ratio'] = (df['A_rt_ratio'] > df['B_rt_ratio']).astype('int')

            df['nf1_diff'] = (df['A_network_feature_1'] > df['B_network_feature_1']).astype('int')
            df['nf2_diff'] = (df['A_network_feature_2'] > df['B_network_feature_2']).astype('int')
            df['nf3_diff'] = (df['A_network_feature_3'] > df['B_network_feature_3']).astype('int')
            
            df['nf3_ratio'] = (df['A_network_feature_3'] / np.maximum(1,df['B_network_feature_3']))
            df['nf2_ratio'] = (df['A_network_feature_2'] / np.maximum(1,df['B_network_feature_2']))
            df['nf1_ratio'] = (df['A_network_feature_1'] / np.maximum(1,df['B_network_feature_1']))

            return df    
    
        
        return feat_eng(pd.read_csv(f'{path}/train.csv').rename({'Choice':'target'}, axis=1)),\
               feat_eng(pd.read_csv(f'{path}/test.csv'))


    return pd.read_csv(f'{path}/train.csv').rename({'Choice':'target'}, axis=1), \
           pd.read_csv(f'{path}/test.csv')

def submit_dataset_predict_who_is_more_influential_in_a_social_network(path, prediction):
    predict_df = pd.read_csv(f'{path}/sample_predictions.csv')[['Id']]
    predict_df['Choice'] = prediction[:,1]

    return predict_df







def load_dataset_santander_customer_satisfaction(path, feature_engineering = False):

    if feature_engineering:
        
        df_train = pd.read_csv(f'{path}/train.csv').drop('ID', axis=1).rename({'TARGET':'target'}, axis=1)

        impute_cols = list(set(df_train.columns) - set(['target']))
        imputer = KNNImputer(n_neighbors=2, weights="uniform").fit(df_train[impute_cols])

        df_train = pd.concat([df_train[[col for col in df_train.columns if col not in impute_cols]],\
                              pd.DataFrame(data=imputer.transform(df_train[impute_cols]), columns=impute_cols)], axis=1)
        df_test = pd.read_csv(f'{path}/test.csv').drop('ID', axis=1)

        df_test = pd.DataFrame(data=imputer.transform(df_test[impute_cols]), columns=impute_cols)

        return df_train, df_test


    return pd.read_csv(f'{path}/train.csv').drop('ID', axis=1).rename({'TARGET':'target'}, axis=1), \
           pd.read_csv(f'{path}/test.csv').drop('ID', axis=1)

def submit_dataset_santander_customer_satisfaction(path, prediction):
    predict_df = pd.read_csv(f'{path}/test.csv')[['ID']]
    predict_df['TARGET'] = prediction[:,1]

    return predict_df







def load_dataset_santander_customer_transaction_prediction(path, feature_engineering = False): 
    if feature_engineering:
        raise "Feature engineering do not implemented"

    return pd.read_csv(f'{path}/train.csv').drop('ID_code', axis=1).rename({'TARGET':'target'}, axis=1), \
           pd.read_csv(f'{path}/test.csv').drop('ID_code', axis=1)

def submit_dataset_santander_customer_transaction_prediction(path, prediction):
    predict_df = pd.read_csv(f'{path}/test.csv')[['ID_code']]
    predict_df['target'] = prediction[:,1]

    return predict_df




competition_dataprep = {'amazon-employee-access-challenge': (load_dataset_amazon_employee, submit_dataset_amazon_employee),
                        'challenges-in-representation-learning-the-black-box-learning-challenge': (load_dataset_challenges_in_representation_learning_the_black_box_learning_challenge, submit_dataset_challenges_in_representation_learning_the_black_box_learning_challenge),
                        'otto-group-product-classification-challenge': (load_dataset_otto_group_product_classification_challenge, submit_dataset_otto_group_product_classification_challenge),
                        'predict-who-is-more-influential-in-a-social-network': (load_dataset_predict_who_is_more_influential_in_a_social_network, submit_dataset_predict_who_is_more_influential_in_a_social_network),
                        'santander-customer-satisfaction': (load_dataset_santander_customer_satisfaction, submit_dataset_santander_customer_satisfaction),
                        'santander-customer-transaction-prediction': (load_dataset_santander_customer_transaction_prediction, submit_dataset_santander_customer_transaction_prediction),
                        }

