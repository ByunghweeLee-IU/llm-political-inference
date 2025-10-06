import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
from collections import defaultdict, Counter
import string
import os, random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib as mpl

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
            plt.close()
            print(f"Saved {png_path} and {pdf_path}")
        except Exception as e:
            print(f"Error creating word cloud for percentile {p}: {e}")
            continue

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
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path} and {pdf_path}")