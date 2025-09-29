import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def get_f1_over_conf(data, confidence_label, prediction_label, true_label):

    '''
    Calculate F1 scores across confidence
    for error bar we use standard error of bootstrap 
    '''
    conf_list = np.arange(1,6) 
    scores = []
    errors = []
    
    party2bin = {"Democratic": 0, "Republican": 1}
    df = data.copy()

    #that fall within an appropriate range.
    df = df[df[prediction_label].isin(party2bin.keys())]
    df = df[df[confidence_label].isin(conf_list)]
    
    # Map labels to binary values
    df['true_party_bin'] = df[true_label].map(party2bin) #0:dem, 1:rep
    df['inferred_party_bin'] = df[prediction_label].map(party2bin) #0:dem, 1:rep
    
    
    for i in conf_list:
        df_ = df[df[confidence_label] == i]
        
        #bootstrap
        bootstrapped_f1_scores = []
        n_size = len(df_)

        for _ in range(1000):
            sample = df_.sample(n=n_size, replace=True) 
            true_labels = sample['true_party_bin']
            pred_labels = sample['inferred_party_bin']
            f1 = f1_score(true_labels, pred_labels, average='macro')
            bootstrapped_f1_scores.append(f1)

        mean_f1 = np.mean(bootstrapped_f1_scores)    
        std = np.std(bootstrapped_f1_scores)    

    
        scores.append(mean_f1)
        errors.append(std)

    return conf_list, scores, errors


def get_conf_distribution(data, confidence_label, prediction_label, true_label):
    '''get the distribution of confidence from a dataset.'''    
    party2bin = {"Democratic": 0, "Republican": 1}
    df = data.copy()

    #that fall within an appropriate range.
    df = df[df[prediction_label].isin(party2bin.keys())]
    df[confidence_label] = df[confidence_label].astype(int) 
    df = df[df[confidence_label].isin(np.arange(1,6))]

    conf_freq = np.arange(5)

    for e in df[confidence_label].values:
        try:
            conf_freq[e-1] += 1
        except:
            print(e)

    return conf_freq / np.sum(conf_freq)
    

#Text-level and User-level inference based on LLM's inference results
def get_text_level_score(df, col1='party_short', col2='party_out_gpt4o'):
    y = df[col1]
    y_hat = df[col2]
    f1 = f1_score(y, y_hat, average='macro')
    #print("Macro F1-score (Text level):", f1)
    return f1
    
def majority_vote(group, prediction_col='party_out_gpt4o'):
    counts = group[prediction_col].value_counts()
    max_count = counts.max()
    top = counts[counts == max_count].index.tolist()
    return np.random.RandomState(seed=42).choice(top)

def weighted_vote(group, prediction_col='party_out_gpt4o', confidence_col='confidence_gpt4o'):
    # Republican=1, Democratic=0
    weight = group[confidence_col]
    binary = group[prediction_col].map({'Republican': 1, 'Democratic': 0})
    avg_score = np.average(binary, weights=weight)
    return 'Republican' if avg_score >= 0.5 else 'Democratic'

def max_confidence_vote(group, prediction_col='party_out_gpt4o', confidence_col='confidence_gpt4o'):
    max_conf = group[confidence_col].max()
    top_rows = group[group[confidence_col] == max_conf]
    # majority -> if tie, choose a random party
    top_preds = top_rows[prediction_col].value_counts()
    max_count = top_preds.max()
    top = top_preds[top_preds == max_count].index.tolist()
    return np.random.RandomState(seed=42).choice(top)




def user_level_f1_fourtypes(data, prediction_col, confidence_col, user_name_col='user_name', true_label_col='party_short'):

    """
    input: data
    output: text-level and user-level F1 scores for four cases (text-level, majority, weighted, max-conf)
    """
    
    #Prepare data 
    df = data.copy()

    ## Filter for valid predictions and confidence scores
    party2bin = {"Democratic": 0, "Republican": 1}
    df = df[df[prediction_col].isin(party2bin.keys())]
    df = df[df[confidence_col].isin([1, 2, 3, 4, 5])]
    
    ## Map labels to binary values
    df['true_party_bin'] = df[true_label_col].map(party2bin)
    df['inferred_party_bin'] = df[prediction_col].map(party2bin)

    # Text-level score 
    text_level_f1 = get_text_level_score(df, col1=true_label_col, col2=prediction_col)

    # User-level 
    # Iterate over each user's data
    y_user = df.groupby('user_name')['party_short'].first()  #true answer 

    # majority    
    y_hat_majority = df.groupby('user_name').apply(lambda x: majority_vote(x, prediction_col), include_groups=False)
    f1_majority = f1_score(y_user, y_hat_majority, average='macro')
    
    y_hat_weighted = df.groupby('user_name').apply(lambda x: weighted_vote(x, prediction_col, confidence_col), include_groups=False)
    f1_weighted = f1_score(y_user, y_hat_weighted, average='macro')
    
    y_hat_max = df.groupby('user_name').apply(lambda x: max_confidence_vote(x, prediction_col, confidence_col), include_groups=False)
    f1_max = f1_score(y_user, y_hat_max, average='macro')

    return text_level_f1, f1_majority, f1_weighted, f1_max