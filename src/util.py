import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import math

import string
import os, random
from tqdm import tqdm
from collections import defaultdict, Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from wordcloud import WordCloud

import matplotlib as mpl
import matplotlib.pyplot as plt



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



## caculation of user level f1 score for each category 
def user_level_category_f1(data, prediction_col, confidence_col, category_label=None, category=False, user_name_col='user_name', true_label_col='party_short'):

    #Prepare data 
    df = data.copy()

    if category:
        df = df[df[category_label]==category]
        
    ## Filter for valid predictions and confidence scores
    party2bin = {"Democratic": 0, "Republican": 1}
    df = df[df[prediction_col].isin(party2bin.keys())]
    df = df[df[confidence_col].isin([1, 2, 3, 4, 5])]
    
    ## Map labels to binary values
    df['true_party_bin'] = df[true_label_col].map(party2bin)
    df['inferred_party_bin'] = df[prediction_col].map(party2bin)

    category_size = len(df)
    
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

    return text_level_f1, f1_majority, f1_weighted, f1_max, category_size


## User participation similarity between categories 
def calculate_jaccard(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Function to calculate Normalized Pointwise Mutual Information (NPMI)
def calculate_pmi(set1, set2, total_users):
    joint_prob = len(set1.intersection(set2)) / total_users
    prob1 = len(set1) / total_users
    prob2 = len(set2) / total_users
    
    # Ensure probabilities are greater than zero to avoid log of zero errors
    if joint_prob > 0 and prob1 > 0 and prob2 > 0:
        return np.log2(joint_prob / (prob1 * prob2))
    else:
        return 0

def calculate_npmi(set1, set2, total_users):
    joint_prob = len(set1.intersection(set2)) / total_users
    prob1 = len(set1) / total_users
    prob2 = len(set2) / total_users
    
    if joint_prob > 0:
        pmi = np.log2(joint_prob / (prob1 * prob2))
        npmi = pmi / -np.log2(joint_prob)
        return npmi
    else:
        return -1




## Representative word analysis 


# 1. Base NLTK stopwords
base_stop_words = set(stopwords.words('english'))

# 2. Custom filler/common words to remove
custom_stop_words = {
    'would','could','people','also','many','should','must','thing','something','everything',
    'anything','make','get','say','said','go','going','come','let','know','want',
    'like','look','way','well','even','much','right','first','one','two','new',
    'old','http','amp','www','com','org','net','https',
    'mr','mrs','ms','dr',
    'yes','no','ok','okay','thanks','thank','please',
    'hi','hello','hey',
    'lot','lots','stuff',
    'today','yesterday','tomorrow',
    'year','years','month','months','day','days','week','weeks',
    'really','actually','probably','maybe','sort','kind','bit',
    'think','thought','feel','felt','see','seen','saw','watch','watching',
    'heard','hear','hearing',
     # Added common verbs with low topical value
    'take','taken','took','taking',
    'give','given','gave','giving',
    'put','puts','putting',
    'find','found','finding',
    'tell','tells','telling','told',
    'work','works','working','worked',
    'use','uses','using','used',
    'try','tries','tried','trying',
    'need','needs','needed','needing',
    'help','helps','helped','helping',
    'show','shows','showed','showing',
    'start','starts','started','starting',
    'stop','stops','stopped','stopping',
    'call','calls','called','calling',
    'move','moves','moved','moving',
    'play','plays','played','playing',
    'run','runs','ran','running',
    'talk','talks','talked','talking'
}

# 3. Punctuation, digits, and single letters
punct_stop = set(string.punctuation)
digit_stop = {str(i) for i in range(0, 100)}
short_fragments = {chr(i) for i in range(ord('a'), ord('z')+1)}

# 4. Merge all stop words
stop_words = base_stop_words.union(custom_stop_words, punct_stop, digit_stop, short_fragments)

# 5. Word filter function
def is_clean_word(word):
    """
    Returns True if word is purely alphabetical and not in stopwords.
    """
    return word.isalpha() and word.lower() not in stop_words

# 6. Tokenizer + filter
def tokenize_and_filter(text):
    """
    Tokenizes the text, lowercases, removes stopwords, punctuation,
    numbers, and short fragments.
    """
    toks = word_tokenize(text.lower())
    return [w for w in toks if is_clean_word(w)]
    
def compute_word_info(df, text_col='text', conf_col='confidence', true_col='true_label', inferred_col='inferred_label'):
    word2info = defaultdict(lambda: {'confidence': [], 'true_label': [], 'inferred_label': []})

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
        toks = tokenize_and_filter(r[text_col])
        conf = r[conf_col]
        tlab = r[true_col]
        ilab = r[inferred_col]

        for w in toks:
            word2info[w]['confidence'].append(conf)
            word2info[w]['true_label'].append(tlab)
            word2info[w]['inferred_label'].append(ilab)

    return word2info


def compute_avg_conf_and_f1(word2info, min_freq=0):
    """
    Filter words by count (> min_freq) first, then compute:
      avg_confidence, count, F1 (macro) using sklearn.

    word2info: dict[word] -> {'confidence': [...], 'true_label': [...], 'inferred_label': [...]}
    returns: dict[word] -> (avg_conf, count, f1_macro)
    """
    result = {}
    for w, info in word2info.items():
        confs = info['confidence']
        n = len(confs)
        if n <= min_freq:      # <-- filter BEFORE F1
            continue

        avg_conf = np.mean(confs) if n else np.nan
        y_true = info['true_label']
        y_pred = info['inferred_label']
        try:
            f1 = f1_score(y_true, y_pred, average='macro')
        except ValueError:
            f1 = np.nan  # e.g., only one class present

        result[w] = (avg_conf, n, f1)
    return result

# ---------------- log odds vs. background ----------------
def log_odds_zscore_vs_background_fast(global_counts: dict, slice_counts: dict, vocab=None, a0=None):
    """
    Fast Monroe-style log-odds + z-score for a restricted vocab.
    - global_counts: dict[word] -> alpha_w  (background counts; Dirichlet prior)
    - slice_counts:  dict[word] -> xw_A     (counts in the slice)
    - vocab: iterable of words to score (defaults to keys of slice_counts)
    - a0: sum of global_counts (pass once from caller for speed)

    Returns DataFrame: word, log_odds_vs_bg, z_score_vs_bg
    """
    if vocab is None:
        vocab = list(slice_counts.keys())

    # vectorize lookups
    alpha_w = np.array([global_counts.get(w, 0.0) for w in vocab], dtype=float)
    xw_A    = np.array([slice_counts.get(w, 0.0)     for w in vocab], dtype=float)

    # totals
    if a0 is None:
        a0 = float(sum(global_counts.values()))
    N_A = float(xw_A.sum())

    # Dirichlet-smoothed odds pieces
    # Guard tiny denominators to avoid inf/nan (doesn't change ranking in practice)
    denom_A = (N_A + a0 - (xw_A + alpha_w))
    denom_bg = (a0 - alpha_w)
    denom_A = np.where(denom_A <= 0, 1e-12, denom_A)
    denom_bg = np.where(denom_bg <= 0, 1e-12, denom_bg)

    num_A  = (xw_A + alpha_w) / denom_A
    num_bg = alpha_w / denom_bg
    odds   = num_A / np.maximum(num_bg, 1e-12)
    log_odds = np.log(np.maximum(odds, 1e-300))

    # Full 4-term SE (Monroe et al., 2008)
    se = np.sqrt(
        1.0 / np.maximum(xw_A + alpha_w, 1e-12) +
        1.0 / np.maximum((N_A - xw_A) + (a0 - alpha_w), 1e-12) +
        1.0 / np.maximum(alpha_w, 1e-12) +
        1.0 / np.maximum(a0 - alpha_w, 1e-12)
    )

    z = log_odds / se

    return pd.DataFrame({
        'word': vocab,
        'log_odds_vs_bg': log_odds,
        'z_score_vs_bg':  z
    })


# --- fast counts-only helper ---
def compute_word_counts(df, text_col='text', desc="Counting words"):
    counts = Counter()
    for _, r in tqdm(df.iterrows(), total=len(df), desc=desc):
        for w in tokenize_and_filter(r[text_col]):
            counts[w] += 1
    return counts

def build_single_confidence_logodds_df(
    df,
    background_df=None,
    party_col='inferred_label',   # used for DEM/REP split that produces freq_dem/freq_rep
    cat_col='category',
    text_col='text',
    conf_col='confidence',
    min_freq=10,
    include_general_slice=True
):
    """
    Returns one DataFrame with:
      category_slice, word, confidence,
      freq_all, freq_dem, freq_rep, freq_dem_true, freq_rep_true,
      f1_all,
      log_odds_all, z_all, log_odds_dem, z_dem, log_odds_rep, z_rep

    Notes:
      - freq_dem/freq_rep come from the *inferred* label split (unchanged behavior).
      - freq_dem_true/freq_rep_true are *additional* columns from the *true* label split.
      - f1_dem/f1_rep are dropped (ambiguous + slow). f1_all is kept.
      - DEM/REP log-odds remain based on the inferred split (as before).
    """
    if background_df is None:
        background_df = df

    # Background α counts from the full corpus (after same cleaning)
    global_counts = Counter()
    for _, row in tqdm(background_df.iterrows(), total=len(background_df), desc="Global background"):
        for w in tokenize_and_filter(row[text_col]):
            global_counts[w] += 1
    a0 = float(sum(global_counts.values()))  # precompute once

    categories = list(df[cat_col].unique())
    if include_general_slice:
        categories += ['general']

    out = []

    for cat in categories:
        sub = df[df[cat_col] != 'Politics'] if cat == 'general' else df[df[cat_col] == cat]

        # ---------- Build "ALL" word info (kept for avg confidence + f1_all + min_freq gate) ----------
        all_w2i = compute_word_info(sub, text_col=text_col, conf_col=conf_col)  # uses true/inferred internally

        # Filter-by-count happens *inside* this call for ALL
        all_stats = compute_avg_conf_and_f1(all_w2i, min_freq=min_freq)

        # Build ALL frame directly from filtered stats (no extra count filter needed later)
        all_df_conf = pd.DataFrame([
            {'word': w, 'confidence': a, 'freq_all': f, 'f1_all': f1}
            for w, (a, f, f1) in all_stats.items()
        ])

        # If nothing passes min_freq in this category, skip
        if all_df_conf.empty:
            continue

        # Keep the same vocab for the rest of the computations
        vocab = set(all_df_conf['word'])

        # ---------- Inferred-label split (drives freq_dem / freq_rep and inferred-based log-odds) ----------
        labels_inf = sub[party_col].astype(str).str.strip().str.lower()
        dem_sub_inf = sub[labels_inf.str.startswith('dem', na=False)]
        rep_sub_inf = sub[labels_inf.str.startswith('rep', na=False)]

        # counts-only for speed
        dem_counts_inf_all = compute_word_counts(dem_sub_inf, text_col, desc="Inferred DEM word counts")
        rep_counts_inf_all = compute_word_counts(rep_sub_inf, text_col, desc="Inferred REP word counts")

        # restrict to vocab
        dem_counts_inf = {w: int(dem_counts_inf_all.get(w, 0)) for w in vocab}
        rep_counts_inf = {w: int(rep_counts_inf_all.get(w, 0)) for w in vocab}

        dem_df_inf = pd.DataFrame({'word': list(vocab),
                                   'freq_dem': [dem_counts_inf[w] for w in vocab]})
        rep_df_inf = pd.DataFrame({'word': list(vocab),
                                   'freq_rep': [rep_counts_inf[w] for w in vocab]})

        # ---------- True-label split (adds freq_dem_true / freq_rep_true; counts-only) ----------
        labels_true = sub['true_label'].astype(str).str.strip().str.lower()
        dem_sub_true = sub[labels_true.str.startswith('dem', na=False)]
        rep_sub_true = sub[labels_true.str.startswith('rep', na=False)]

        dem_counts_true_all = compute_word_counts(dem_sub_true, text_col, desc="True DEM word counts")
        rep_counts_true_all = compute_word_counts(rep_sub_true, text_col, desc="True REP word counts")

        dem_counts_true = {w: int(dem_counts_true_all.get(w, 0)) for w in vocab}
        rep_counts_true = {w: int(rep_counts_true_all.get(w, 0)) for w in vocab}

        dem_df_true = pd.DataFrame({'word': list(vocab),
                                    'freq_dem_true': [dem_counts_true[w] for w in vocab]})
        rep_df_true = pd.DataFrame({'word': list(vocab),
                                    'freq_rep_true': [rep_counts_true[w] for w in vocab]})

        # ---------- Merge everything ----------
        merged = (all_df_conf
                  .merge(dem_df_inf,  on='word', how='left')
                  .merge(rep_df_inf,  on='word', how='left')
                  .merge(dem_df_true, on='word', how='left')
                  .merge(rep_df_true, on='word', how='left')
                  .fillna({'freq_dem': 0, 'freq_rep': 0, 'freq_dem_true': 0, 'freq_rep_true': 0}))

        # counts & restricted vocab (only kept words)
        vocab_list = merged['word'].tolist()
        all_counts = dict(zip(vocab_list, merged['freq_all'].astype(int)))
        dem_counts = dict(zip(vocab_list, merged['freq_dem'].astype(int)))  # inferred split
        rep_counts = dict(zip(vocab_list, merged['freq_rep'].astype(int)))  # inferred split

        # ---------- FAST log-odds + z for ALL/DEM/REP over vocab (DEM/REP based on inferred split) ----------
        stats_all = log_odds_zscore_vs_background_fast(global_counts, all_counts, vocab=vocab_list, a0=a0)\
                        .rename(columns={'log_odds_vs_bg':'log_odds_all','z_score_vs_bg':'z_all'})
        stats_dem = log_odds_zscore_vs_background_fast(global_counts, dem_counts, vocab=vocab_list, a0=a0)\
                        .rename(columns={'log_odds_vs_bg':'log_odds_dem','z_score_vs_bg':'z_dem'})
        stats_rep = log_odds_zscore_vs_background_fast(global_counts, rep_counts, vocab=vocab_list, a0=a0)\
                        .rename(columns={'log_odds_vs_bg':'log_odds_rep','z_score_vs_bg':'z_rep'})

        merged = (merged
                  .merge(stats_all, on='word', how='left')
                  .merge(stats_dem, on='word', how='left')
                  .merge(stats_rep, on='word', how='left'))

        merged['category_slice'] = cat
        merged = merged[
            ['category_slice','word','confidence',
             'freq_all','freq_dem','freq_rep','freq_dem_true','freq_rep_true',
             'f1_all',
             'log_odds_all','z_all','log_odds_dem','z_dem','log_odds_rep','z_rep']
        ].sort_values(['category_slice','confidence'], ascending=[True, False]).reset_index(drop=True)

        out.append(merged)

    if not out:
        # If no category produced rows (e.g., too strict min_freq), return empty frame with expected columns
        return pd.DataFrame(columns=[
            'category_slice','word','confidence',
            'freq_all','freq_dem','freq_rep','freq_dem_true','freq_rep_true',
            'f1_all',
            'log_odds_all','z_all','log_odds_dem','z_dem','log_odds_rep','z_rep'
        ])

    return pd.concat(out, ignore_index=True)

def compute_leaning_per_category(
    df,
    cat_col="category_slice",
    word_col="word",
    rep_count_col="freq_rep",
    dem_count_col="freq_dem",
    neutral_value=0.5,   # used only when denominator is 0
    on_zero="neutral",   # "neutral" -> assign neutral_value; "skip" -> omit those words
):
    """
    NO SMOOTHING.
      p_rep(word) = rep_count(word) / sum_rep_counts_in_category
      p_dem(word) = dem_count(word) / sum_dem_counts_in_category
      leaning = p_rep / (p_rep + p_dem)  in [0,1]
    Returns {(category, word): leaning}.
    """
    out = {}
    for cat, g in df.dropna(subset=[cat_col]).groupby(cat_col):
        rep_tot = g[rep_count_col].sum()
        dem_tot = g[dem_count_col].sum()

        p_rep = g[rep_count_col] / rep_tot if rep_tot > 0 else 0.0
        p_dem = g[dem_count_col] / dem_tot if dem_tot > 0 else 0.0

        denom = (p_rep + p_dem).to_numpy()
        numer = p_rep.to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            lean = np.where(denom > 0, numer / denom,
                            (np.nan if on_zero == "skip" else neutral_value))

        for w, l in zip(g[word_col].to_numpy(), lean):
            if np.isnan(l):
                if on_zero == "skip":
                    continue
                l = neutral_value
            out[(cat, w)] = float(l)
    return out

def make_party_gradient_color_func(
    word2leaning_for_cat,
    blue_rgb=(60, 90, 220),
    gray_rgb=(128, 128, 128),
    red_rgb=(220, 70, 60),
    mid=0.5,
    mid_width=0.10,   # e.g., [0.45, 0.55] is gray
    edge_gamma=0.75,  # <1 expands colors near the edges of the gray band
    jitter=0,
):
    """
    Blue ←—— Gray ——→ Red with a narrow gray plateau.
    - mid_width: width of pure-gray zone centered at 'mid'.
    - edge_gamma: nonlinearity outside the gray zone (t^edge_gamma). <1 pulls faster toward ends.
    """
    br, bg, bb = blue_rgb
    gr, gg, gb = gray_rgb
    rr, rg, rb = red_rgb

    lo = max(0.0, mid - mid_width/2.0)  # start of gray zone
    hi = min(1.0, mid + mid_width/2.0)  # end of gray zone

    def _clip(x, lo=0, hi=255):
        return max(lo, min(hi, int(round(x))))

    def _lerp(a, b, t):
        return a + (b - a) * t

    def _color(word, font_size, position, orientation, random_state=None, **kwargs):
        l = float(word2leaning_for_cat.get(word, mid))

        # Optional jitter
        j = (random_state.randint(-jitter, jitter+1) if (random_state and jitter) else 0)

        if l <= lo:
            # 0..lo  → 0..1  (Blue → Gray), with non-linear ramp
            t = (l - 0.0) / (lo - 0.0) if lo > 0 else 1.0
            t = t ** edge_gamma
            r = _clip(_lerp(br, gr, t) + j); g = _clip(_lerp(bg, gg, t) + j); b = _clip(_lerp(bb, gb, t) + j)
        elif l >= hi:
            # hi..1  → 0..1  (Gray → Red), with non-linear ramp
            t = (l - hi) / (1.0 - hi) if hi < 1 else 1.0
            t = t ** edge_gamma
            r = _clip(_lerp(gr, rr, t) + j); g = _clip(_lerp(gg, rg, t) + j); b = _clip(_lerp(gb, rb, t) + j)
        else:
            # Inside gray plateau
            r = gr + j; g = gg + j; b = gb + j
            r = _clip(r); g = _clip(g); b = _clip(b)

        return f"rgb({r},{g},{b})"

    return _color

def generate_wordclouds_by_category(
    df: pd.DataFrame,
    value_col: str = "log_odds_all",
    conf_col: str = "confidence",
    cat_col: str = "category_slice",
    word_col: str = "word",
    # party count columns for leaning
    rep_count_col: str = "freq_rep",
    dem_count_col: str = "freq_dem",
    on_zero: str = "neutral",      # "skip" to drop words when denom==0
    neutral_value: float = 0.5,    # used only if on_zero == "neutral"
    # color settings
    blue_rgb=(60, 90, 220),
    gray_rgb=(128, 128, 128),
    red_rgb=(220, 70, 60),
    mid: float = 0.5,              # center of gray band
    mid_width: float = 0.10,       # e.g., 0.10 → [0.45,0.55] gray
    edge_gamma: float = 0.75,      # <1 intensifies color just outside gray band
    jitter: int = 0,
    # I/O and misc
    output_dir: str = "wordclouds",
    width: int = 1000,
    height: int = 1000,
    dpi: int = 300,
    random_state: int = 42,
    top_k_if_large: int = 100,
):
    """
    Creates 5 wordclouds per category (quintiles of `conf_col`).
      - Word size = `value_col` values.
      - Word color = Blue ← Gray → Red using category-wise leaning from raw counts.
      - ALWAYS cap each bin to top `top_k_if_large` words (if available).
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_state)

    # Precompute (category, word) → leaning in [0,1]
    catword2lean = compute_leaning_per_category(
        df,
        cat_col=cat_col, word_col=word_col,
        rep_count_col=rep_count_col, dem_count_col=dem_count_col,
        neutral_value=neutral_value, on_zero=on_zero,
    )

    for cat in df[cat_col].dropna().unique().tolist():
        sub = df[df[cat_col] == cat].copy()
        sub["percentile"] = pd.cut(sub[conf_col].rank(method='average'), bins=5, labels=False, include_lowest=True)

        # Build color function for this category
        word2leaning_this_cat = {
            w: catword2lean.get((cat, w), neutral_value)
            for w in sub[word_col].unique()
        }

        color_func = make_party_gradient_color_func(
            word2leaning_for_cat=word2leaning_this_cat,
            blue_rgb=blue_rgb,
            gray_rgb=gray_rgb,
            red_rgb=red_rgb,
            mid=mid,
            mid_width=mid_width,
            edge_gamma=edge_gamma,
            jitter=jitter,
        )

        # Force 5 bins
        for p in range(5):
            g = sub[sub["percentile"] == p].copy()
            # No need to filter negatives - already done in data preparation!

            # Cap to top-K
            if len(g) > top_k_if_large:
                g = (
                    g.sort_values(by=[value_col, word_col],
                                  ascending=[False, True])
                     .head(top_k_if_large)
                )

            freq_dict = dict(zip(g[word_col], g[value_col]))
            if not freq_dict:
                continue

            try:
                wc = WordCloud(
                    width=width,
                    height=height,
                    background_color="white",
                    random_state=random_state
                ).generate_from_frequencies(freq_dict)

                wc.recolor(color_func=color_func)

                base = f"{cat}_perc{int(p)}_top{len(freq_dict)}_by_{value_col}"
                png_path = os.path.join(output_dir, base + ".png")
                pdf_path = os.path.join(output_dir, base + ".pdf")

                plt.figure(figsize=(width/100, height/100), dpi=dpi)
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
                plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
                plt.close()
                print(f"Saved {png_path} and {pdf_path}")
            except Exception as e:
                print(f"Error creating word cloud for {cat} percentile {p}: {e}")
                continue

def generate_wordclouds_for_category(
    df: pd.DataFrame,
    category: str,
    value_col: str = "log_odds_all",
    conf_col: str = "confidence",
    cat_col: str = "category_slice",
    word_col: str = "word",
    # party count columns for leaning
    rep_count_col: str = "freq_rep",
    dem_count_col: str = "freq_dem",
    on_zero: str = "neutral",      # "skip" to drop words when denom==0
    neutral_value: float = 0.5,    # used only if on_zero == "neutral"
    # color settings
    blue_rgb=(60, 90, 220),
    gray_rgb=(128, 128, 128),
    red_rgb=(220, 70, 60),
    mid: float = 0.5,              # center of gray band
    mid_width: float = 0.10,       # e.g., 0.10 → [0.45,0.55] gray
    edge_gamma: float = 0.75,      # <1 intensifies color just outside gray band
    jitter: int = 0,
    # I/O and misc
    output_dir: str = "wordclouds_single_category",
    width: int = 1000,
    height: int = 1000,
    dpi: int = 300,
    random_state: int = 42,
    top_k_if_large: int = 100,
):
    """
    Creates 5 wordclouds for one category (quintiles of `conf_col`).
      - Word size = `value_col` values.
      - Word color = Blue ← Gray → Red using category-wise leaning from raw counts.
      - ALWAYS cap each bin to top `top_k_if_large` words (if available).
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_state)

    sub = df[df[cat_col] == category].copy()
    if sub.empty:
        print(f"[WARN] No rows for category='{category}'. Skipping.")
        return

    # Precompute (word) → leaning in [0,1] for this category
    catword2lean = compute_leaning_per_category(
        sub,
        cat_col=cat_col, word_col=word_col,
        rep_count_col=rep_count_col, dem_count_col=dem_count_col,
        neutral_value=neutral_value, on_zero=on_zero,
    )
    word2leaning_this_cat = {
        w: catword2lean.get((category, w), neutral_value)
        for w in sub[word_col].unique()
    }

    color_func = make_party_gradient_color_func(
        word2leaning_for_cat=word2leaning_this_cat,
        blue_rgb=blue_rgb,
        gray_rgb=gray_rgb,
        red_rgb=red_rgb,
        mid=mid,
        mid_width=mid_width,
        edge_gamma=edge_gamma,
        jitter=jitter,
    )

    # Bin by confidence (always integers 0..4)
    sub["percentile"] = pd.cut(sub[conf_col].rank(method='average'), bins=5, labels=False, include_lowest=True)

    for p in range(5):
        g = sub[sub["percentile"] == p].copy()
        # No need to filter negatives - already done in data preparation!

        # Cap to top-K
        if len(g) > top_k_if_large:
            g = (
                g.sort_values(by=[value_col, word_col],
                              ascending=[False, True])
                 .head(top_k_if_large)
            )

        freq_dict = dict(zip(g[word_col], g[value_col]))
        if not freq_dict:
            continue

        try:
            wc = WordCloud(
                width=width,
                height=height,
                background_color="white",
                random_state=random_state
            ).generate_from_frequencies(freq_dict)

            wc.recolor(color_func=color_func)

            base = f"{category}_perc{int(p)}_top{len(freq_dict)}_by_{value_col}"
            png_path = os.path.join(output_dir, base + ".png")
            pdf_path = os.path.join(output_dir, base + ".pdf")

            plt.figure(figsize=(width/100, height/100), dpi=dpi)
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
            plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
            plt.show()
            #plt.close()
            print(f"Saved {png_path} and {pdf_path}")
        except Exception as e:
            print(f"Error creating word cloud for percentile {p}: {e}")
            continue



def generate_wordclouds_for_category_5panels(
    df: pd.DataFrame,
    category: str,
    value_col: str = "log_odds_all",
    conf_col: str = "confidence",
    cat_col: str = "category_slice",
    word_col: str = "word",
    rep_count_col: str = "freq_rep",
    dem_count_col: str = "freq_dem",
    on_zero: str = "neutral",
    neutral_value: float = 0.5,
    blue_rgb=(60, 90, 220),
    gray_rgb=(128, 128, 128),
    red_rgb=(220, 70, 60),
    mid: float = 0.5,
    mid_width: float = 0.10,
    edge_gamma: float = 0.75,
    jitter: int = 0,
    output_dir: str = "wordclouds_single_category",
    width: int = 1000,
    height: int = 1000,
    dpi: int = 300,
    random_state: int = 42,
    top_k_if_large: int = 100,
    show_inline: bool = True,    # whether to show in the notebook or not 
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_state)

    sub = df[df[cat_col] == category].copy()
    if sub.empty:
        print(f"[WARN] No rows for category='{category}'. Skipping.")
        return

    # Precompute leaning 
    catword2lean = compute_leaning_per_category(
        sub,
        cat_col=cat_col, word_col=word_col,
        rep_count_col=rep_count_col, dem_count_col=dem_count_col,
        neutral_value=neutral_value, on_zero=on_zero,
    )
    word2leaning_this_cat = {
        w: catword2lean.get((category, w), neutral_value)
        for w in sub[word_col].unique()
    }

    color_func = make_party_gradient_color_func(
        word2leaning_for_cat=word2leaning_this_cat,
        blue_rgb=blue_rgb, gray_rgb=gray_rgb, red_rgb=red_rgb,
        mid=mid, mid_width=mid_width, edge_gamma=edge_gamma, jitter=jitter,
    )

    sub["percentile"] = pd.cut(sub[conf_col].rank(method='average'),
                               bins=5, labels=False, include_lowest=True)

    # 5 subplots (1row 5col)
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # 💡 size 
    fig.suptitle(f"Word Clouds for {category}", fontsize=18)

    for p in range(5):
        g = sub[sub["percentile"] == p].copy()
        if len(g) > top_k_if_large:
            g = g.sort_values(by=[value_col, word_col],
                              ascending=[False, True]).head(top_k_if_large)

        freq_dict = dict(zip(g[word_col], g[value_col]))
        ax = axes[p]
        ax.axis("off")
        if not freq_dict:
            ax.text(0.5, 0.5, f"Empty (bin {p})", ha="center", va="center")
            continue

        wc = WordCloud(
            width=width//2,
            height=height//2,
            background_color="white",
            random_state=random_state
        ).generate_from_frequencies(freq_dict)
        wc.recolor(color_func=color_func)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Q{p+1}")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{category}_5panels.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {save_path}")
    if show_inline:
        plt.show()
    plt.close()

#color bar
def save_party_colorbar(
    output_dir="wordclouds_single_category_zscore",
    filename="colorbar_bgr",
    *,
    # colors
    blue_rgb=(60, 90, 220),
    gray_rgb=(128, 128, 128),
    red_rgb=(220, 70, 60),
    # mapping controls
    mid=0.5,                 # center of gray (0..1)
    mid_width=0.0,           # width of gray plateau around `mid`; 0 => no plateau
    edge_gamma=0.7,          # <1 pulls faster toward blue/red; >1 gentler
    # figure & labels
    figsize=(1.2, 5.0),
    title="Word leaning\n(0 = Dem, 1 = Rep)",
    ticks=(0, 0.25, 0.5, 0.75, 1.0),
    dpi=300,
):
    """
    Save a vertical color bar showing the Blue ← Gray → Red mapping with center-aware
    nonlinearity and optional gray plateau.

    Parameters
    ----------
    mid : float
        Midpoint in [0,1] that maps to gray.
    mid_width : float
        Width of a flat gray band centered at `mid`. Use 0 for a continuous blend.
    edge_gamma : float
        Exponent applied to the normalized distance from the midpoint on each side.
        Values <1 increase color intensity near the midpoint; >1 softens it.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize RGB to 0..1
    b = np.array(blue_rgb, dtype=float) / 255.0
    g = np.array(gray_rgb, dtype=float) / 255.0
    r = np.array(red_rgb, dtype=float) / 255.0

    # Build colormap stops
    mid_lo = max(0.0, float(mid) - float(mid_width) / 2.0)
    mid_hi = min(1.0, float(mid) + float(mid_width) / 2.0)
    if mid_width > 0:
        # Blue → Gray plateau → Red
        stops = [(0.0, b), (mid_lo, g), (mid_hi, g), (1.0, r)]
    else:
        # Continuous Blue → Gray → Red
        stops = [(0.0, b), (float(mid), g), (1.0, r)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("BlueGrayRed", stops)

    class CenterGammaNorm(mpl.colors.Normalize):
        """Piecewise nonlinearity centered at `mid`, with optional gray plateau (no NaN warnings)."""
        def __init__(self, vmin=0.0, vmax=1.0, clip=True, mid=0.5, mid_width=0.0, edge_gamma=1.0):
            super().__init__(vmin=vmin, vmax=vmax, clip=clip)
            self.mid = float(mid)
            self.mid_width = float(mid_width)
            self.edge_gamma = float(edge_gamma)

        def __call__(self, value, clip=None):
            x = np.asarray(value, dtype=float)
            # normalize to [0,1]
            x = (x - self.vmin) / (self.vmax - self.vmin)
            x = np.clip(x, 0.0, 1.0)

            m  = self.mid
            w  = self.mid_width
            g  = self.edge_gamma
            lo = max(0.0, m - w / 2.0)
            hi = min(1.0, m + w / 2.0)

            y = np.empty_like(x)

            if w > 0.0:
                # Masks
                left = x <= lo
                midx = (x > lo) & (x < hi)
                rght = x >= hi

                # Left branch: [0, lo] → [0, 0.5]
                if lo > 0:
                    t = np.clip(x[left] / lo, 0.0, 1.0)
                    y[left] = 0.5 * np.power(t, g)
                else:
                    y[left] = 0.0

                # Mid plateau
                y[midx] = 0.5

                # Right branch: [hi, 1] → [0.5, 1]
                if hi < 1:
                    t = np.clip((x[rght] - hi) / (1.0 - hi), 0.0, 1.0)
                    y[rght] = 0.5 + 0.5 * np.power(t, g)
                else:
                    y[rght] = 1.0

            else:
                # Continuous (no plateau): split exactly at m
                left = x <= m
                rght = x > m

                if m > 0:
                    t = np.clip(x[left] / m, 0.0, 1.0)
                    y[left] = 0.5 * np.power(t, g)
                else:
                    # m == 0 ⇒ everything is effectively on the right
                    y[left] = 0.0

                if (1.0 - m) > 0:
                    t = np.clip((x[rght] - m) / (1.0 - m), 0.0, 1.0)
                    y[rght] = 0.5 + 0.5 * np.power(t, g)
                else:
                    # m == 1 ⇒ everything is effectively on the left
                    y[rght] = 1.0

            return y

    norm = CenterGammaNorm(vmin=0.0, vmax=1.0, mid=mid, mid_width=mid_width, edge_gamma=edge_gamma)

    fig, ax = plt.subplots(figsize=figsize)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label(title, fontsize=10)
    cb.set_ticks(list(ticks))
    cb.set_ticklabels([f"{t:g}" for t in ticks])

    png_path = os.path.join(output_dir, f"{filename}.png")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    #plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    #plt.close()
    print(f"Saved {png_path} and {pdf_path}")

    


## word confidence vs f1 score (Fig 4c,d)
def plot_conf_vs_f1(
    df, title, save_path=None,
    n_bins=15,                  # quantile bins  
    min_per_bin=200,            # 
    label=None,
    color='#932DE7'
):
    d = df.dropna(subset=["confidence", "f1_all"]).copy()
    d = d[(d["confidence"] >= 1) & (d["confidence"] <= 5)]

    
    # 
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(d["confidence"], q)
    # 
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError("Quantile edges collapsed")
    
    centers = (edges[:-1] + edges[1:]) / 2
    d["conf_bin"] = pd.cut(d["confidence"], bins=edges, include_lowest=True)

    grp = d.groupby("conf_bin")["f1_all"]
    mean_vals = grp.mean().to_numpy()
    std_vals  = grp.std().to_numpy()
    counts    = grp.count().to_numpy()
    ci95 = 1.96 * (std_vals / np.sqrt(np.maximum(counts, 1)))

    valid = ~np.isnan(mean_vals)
    x = centers[valid]
    y = mean_vals[valid]
    y_lower = y - ci95[valid]
    y_upper = y + ci95[valid]


    #plot
    fig, ax = plt.subplots(figsize=(3.9, 3.4), dpi=300)
    ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)
    ax.plot(x, y, marker="o", linewidth=2, markersize=6, color=color, label=label)
    ax.axhline(0.5, ls='--', color='gray')

    ax.set_xlabel("Word-level confidence", fontsize=15)
    ax.set_ylabel("Mean F1 score", fontsize=15)
    ax.set_xlim(1, 5)
    ax.set_xticks(np.linspace(1, 5, 5))
    if label:
        ax.legend(frameon=False, fontsize=12,loc=2)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_ylim(min(0.45, min(y_lower)-0.02), max(y_upper)+0.02)
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches="tight")
    plt.show()



def _conf_f1_curve(d, n_bins=20):
    d = d.dropna(subset=["confidence", "f1_all"]).copy()
    d = d[(d["confidence"] >= 1) & (d["confidence"] <= 5)]
    if len(d) < 3:
        return None  


    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(d["confidence"], q)
    edges = np.unique(edges)
    if len(edges) < 2:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    d["conf_bin"] = pd.cut(d["confidence"], bins=edges, include_lowest=True)

    grp = d.groupby("conf_bin", observed=False)["f1_all"]
    mean_vals = grp.mean().to_numpy()
    std_vals  = grp.std().to_numpy()
    counts    = grp.count().to_numpy()
    ci95 = 1.96 * (std_vals / np.sqrt(np.maximum(counts, 1)))

    valid = ~np.isnan(mean_vals)
    x = centers[valid]
    y = mean_vals[valid]
    y_lower = y - ci95[valid]
    y_upper = y + ci95[valid]
    return x, y, y_lower, y_upper



# ---- multiplot for word confidence vs f1 scores----
def multiplot_conf_vs_f1(
    df_all, category_col="category_slice", n_bins=15, 
    ncols=6, ymin=0.4, ymax=0.95, save_path=None, color='#285E3D'):
    
    cats = sorted(df_all[category_col].dropna().unique())
    n = len(cats)
    nrows = math.ceil(n / ncols)

    # 
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.2, nrows*2.2), dpi=300, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for i, c in enumerate(cats):
        r, k = divmod(i, ncols)
        ax = axes[r, k]
        d = df_all[df_all[category_col] == c]

        curve = _conf_f1_curve(d, n_bins=n_bins)
        if curve is None:
            ax.set_title(f"{c} (no data)", fontsize=9)
            ax.set_xlim(1, 5); ax.set_ylim(ymin, ymax)
            ax.axhline(0.5, ls="--", color="gray", linewidth=0.8)
            ax.tick_params(labelsize=8)
            continue

        x, y, y_lo, y_hi = curve

        ax.fill_between(x, y_lo, y_hi, alpha=0.2, color=color)     
        ax.plot(x, y, marker="o", color=color, linewidth=1.5, markersize=3)

        ax.axhline(0.5, ls="--", color="gray", linewidth=0.8)
        ax.set_xlim(1, 5); ax.set_ylim(ymin, ymax)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=8)

        #
        if r < nrows - 1:
            ax.set_xlabel("")
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Word-level Confidence", fontsize=9)
        if k > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Mean F1 score", fontsize=9)

    # 
    for j in range(n, nrows*ncols):
        r, k = divmod(j, ncols)
        axes[r, k].axis("off")

    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    if save_path:
        fig.savefig(save_path, transparent=True, bbox_inches="tight")
    plt.show()

    

