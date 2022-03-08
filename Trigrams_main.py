import pandas as pd
import numpy as np
import os

#this block is used to generate trigrams,bigrams and unigrams for multiple use cases in the rest of the code
def generate_n_grams(text, ngram):
    words = [''.join(char for char in word if char.isalnum()).lower() for word in text.split(" ")]
    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans

#used to calculate counts and probabilities for ngrams when given tokens
def getCountsandProbability(tokens, n_gram, smoothing=0):
    counts, probability, curr_corpus, prev_corpus = [],[],[],[]
    if n_gram == 3:
        curr_corpus = trigrams_corpus
        prev_corpus = bigram_corpus
    elif n_gram == 2:
        curr_corpus = bigram_corpus
        prev_corpus = unigram_corpus
    elif n_gram == 1:
        curr_corpus = unigram_corpus
    for token in tokens:
        tg_count = curr_corpus.count(token) + smoothing
        counts.append(tg_count)
        if n_gram == 1:
            bg_count = len(tokenized_corpus)
        else:
            bg_count = prev_corpus.count(" ".join(token.split(" ")[:n_gram - 1])) + (0 if smoothing == 0 else smoothing * len(vocab))
        probability.append(tg_count / bg_count)
    return counts, probability


#calculating reconstituted counts
def counts_reconsttd(trigram, resp):
    trigram_words=" ".join(trigram.split(" ")[:2])
    counters=bigram_corpus.count(trigram_words)
    recon_cnt=resp * counters
    return recon_cnt


def create_counts_prob(cmt, ngram):
    ngrams_s = generate_n_grams(cmt, ngram)
    table_df = pd.DataFrame()
    table_df[NGRAMS] = ngrams_s
    table_df[CNT_unsm], table_df[PROB_unsm] = getCountsandProbability(ngrams_s, ngram)
    table_df[CNT_sm], table_df[PROB_sm] = getCountsandProbability(ngrams_s, ngram, 1)
    table_df[CNT_reconstd] = [counts_reconsttd(trigram, prob) for trigram, prob in
                                zip(table_df[NGRAMS], table_df[PROB_sm])]
    table_df[PROB_dsc] = table_df[CNT_reconstd] / table_df[CNT_unsm]
    table_df[PROB_dsc].replace(np.inf, 0, inplace=True)
    table_df[PROB_dsc].replace(np.nan, 0, inplace=True)
    return table_df


#Calculating Katz Back off Probabilities
def Katz_backoff_prob(token, c, dist_prob, ngram, counts_df, bi_g, uni_g, flag_1):
    global trigrams_s1_cnpb, bigrams_s1_cnpb, unigram_s1_cnpb, trigram_s2_cnpb, biigram_s2_cnpb, unigram_s2_cnpb
    token = " ".join(token.split(" ")[:ngram - 1])
    if c > 0:
        if flag_1:
            trigrams_s1_cnpb += 1
        else:
            trigram_s2_cnpb += 1
        return dist_prob
    elif ngram == 3:
        if flag_1:
            bigrams_s1_cnpb += 1
        else:
            biigram_s2_cnpb += 1
        dataframe_tp = bi_g[bi_g[NGRAMS] == token]
        new_c = 0 if dataframe_tp[CNT_unsm].empty else dataframe_tp[CNT_unsm].iloc[0]
        new_dist_prob = 0 if dataframe_tp[PROB_dsc].empty else dataframe_tp[PROB_dsc].iloc[0]
        dist_prob_num = counts_df[counts_df[CNT_unsm] > 0][PROB_dsc]
        dist_prob_den = dataframe_tp[dataframe_tp[CNT_unsm] > 0][PROB_dsc]
        return alpha_apply(dist_prob_num, dist_prob_den) * Katz_backoff_prob(token, new_c, new_dist_prob, ngram - 1, dataframe_tp, bi_g, uni_g, flag_1)
    elif ngram == 2:
        if flag_1:
            unigram_s1_cnpb += 1
        else:
            unigram_s2_cnpb += 1
        dataframe_tp = uni_g[uni_g[NGRAMS] == token]
        new_c = 0 if dataframe_tp[CNT_unsm].empty else dataframe_tp[CNT_unsm].iloc[0]
        new_dist_prob = 0 if dataframe_tp[PROB_dsc].empty else dataframe_tp[PROB_dsc].iloc[0]
        dist_prob_num = counts_df[counts_df[CNT_unsm] > 0][PROB_dsc]
        dist_prob_den = dataframe_tp[dataframe_tp[CNT_unsm] > 0][PROB_dsc]
        return alpha_apply(dist_prob_num, dist_prob_den) * Katz_backoff_prob(token, new_c, new_dist_prob, ngram - 1, dataframe_tp, bi_g, uni_g, flag_1)
    dist_prob_num = counts_df[counts_df[CNT_unsm] > 0][PROB_dsc]
    dist_prob_den = pd.Series([-1])
    return alpha_apply(dist_prob_num, dist_prob_den)


##function to calculate and apply alpha on backtracking probability
def alpha_apply(dp_num, dp_den):
    if len(dp_den) == 1 and dp_den.iloc[0] == -1:
        return (1 - dp_num.sum())
    return (1 - dp_num.sum()) / (1 - dp_den.sum())



NGRAMS = "ngrams"
CNT_unsm = "CNT_unsm"
PROB_unsm = "PROB_unsm"
CNT_sm = "CNT_sm"
PROB_sm = "PROB_sm"
CNT_reconstd = "CNT_reconstd"
PROB_dsc = "PROB_dsc"
PROB_katz = "katz_prob"

s1 = "Sales of the company to return to normalcy."
s2 = "The new products and services contributed to increase revenue."
tokenized_corpus = []
vocab = []


#initialize 0..nth grams
trigrams_corpus = []
bigram_corpus = []
unigram_corpus = []

file = os.path.join("corpus_for_language_models.txt")
with open(file) as f:
    contents = f.read().split()
    for content in contents:
        temp = ''.join(char for char in content if char.isalnum()).lower()
        if temp != '':
            tokenized_corpus.append(temp)

vocab = set(tokenized_corpus)
trigrams_corpus = generate_n_grams(" ".join(tokenized_corpus), 3)
bigram_corpus = generate_n_grams(" ".join(tokenized_corpus), 2)
unigram_corpus = generate_n_grams(" ".join(tokenized_corpus), 1)


trigrams_s1 = []
trigrams_s2 = []

# Write a program to compute the trigrams for any given input. TOTAL: 2 points
# Apply your program to compute the trigrams you need for sentences S1 and S2.

print("Q1 A) Subpart 1..........................................................................................\n")
trigrams_s1 = generate_n_grams(s1, 3)
trigrams_s2 = generate_n_grams(s2, 3)
print("Trigram S1: ", trigrams_s1)
print("Trigram S2: ", trigrams_s2,"\n")

trigrams_s1_cdp = pd.DataFrame()
trigrams_s2_cdp = pd.DataFrame()


# Construct automatically (by the program) the tables with (a) the trigram counts (2 points)
# and the (b) trigram probabilities for the language model without smoothing. (3 points) TOTAL: 5 points
print("Q1 A) Subpart 2.............................................................................................\n")
trigrams_s1_cdp[NGRAMS] = trigrams_s1
trigrams_s2_cdp[NGRAMS] = trigrams_s2
trigrams_s1_cdp[CNT_unsm], trigrams_s1_cdp[PROB_unsm] = getCountsandProbability(trigrams_s1, 3)
trigrams_s2_cdp[CNT_unsm], trigrams_s2_cdp[PROB_unsm] = getCountsandProbability(trigrams_s2, 3)
print("(a) the trigram counts(2 points)\n")
print("S1: \n", trigrams_s1_cdp[[NGRAMS, CNT_unsm]],"\n")
print("S2: \n", trigrams_s2_cdp[[NGRAMS, CNT_unsm]],"\n")
print("(b) trigram probabilities for the language model without smoothing. (3 points)\n")
print("S1: \n", trigrams_s1_cdp[[NGRAMS, PROB_unsm]],"\n")
print("S2: \n", trigrams_s2_cdp[[NGRAMS, PROB_unsm]],"\n")




#
# Construct automatically (by the program): (i) the Laplace-smoothed count tables; (2 points)
# (ii) the Laplace-smoothed probability tables (3 points); and
# (iii) the corresponding re-constituted counts (3 points) TOTAL: 8 points
#
print("Q1 A) Subpart 3................................................................................................")
trigrams_s1_cdp[CNT_sm], trigrams_s1_cdp[PROB_sm] = getCountsandProbability(trigrams_s1, 3, 1)
trigrams_s2_cdp[CNT_sm], trigrams_s2_cdp[PROB_sm] = getCountsandProbability(trigrams_s2, 3, 1)

print("(i) the Laplace-smoothed count tables (2 points)\n")
print("S1: ", trigrams_s1_cdp[[NGRAMS, CNT_sm]],"\n")
print("S2: ", trigrams_s2_cdp[[NGRAMS, CNT_sm]],"\n")

print("(ii) the Laplace-smoothed probability tables (3 points)\n")
print("S1: ",trigrams_s1_cdp[[NGRAMS, PROB_sm]],"\n")
print("S2  ", trigrams_s2_cdp[[NGRAMS, PROB_sm]],"\n")

print("(iii) the corresponding re-constituted counts (3 points)\n")
trigrams_s1_cdp[CNT_reconstd] = [counts_reconsttd(trigram, prob) for trigram, prob in
                                  zip(trigrams_s1_cdp[NGRAMS], trigrams_s1_cdp[PROB_sm])]
print("S1: ", trigrams_s1_cdp[[NGRAMS, CNT_reconstd]],"\n")

trigrams_s2_cdp[CNT_reconstd] = [counts_reconsttd(trigram, prob) for trigram, prob in
                                  zip(trigrams_s2_cdp[NGRAMS], trigrams_s2_cdp[PROB_sm])]
print("S2: ",trigrams_s2_cdp[[NGRAMS, CNT_reconstd]],"\n")


#
# Construct automatically (by the program) the smoothed trigram probabilities using the Katz back-off method. (8 points)
# How many times you had  to also compute the smoothed trigram probabilities (2 points) and
# how many times you had to compute the smoothed unigram probabilities (2 points). TOTAL: 12 points
#

bigram_s1_details,bigram_s2_details,unigram_s1_details,unigram_s2_details = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
trigrams_s1_cnpb,bigrams_s1_cnpb,unigram_s1_cnpb,trigram_s2_cnpb,biigram_s2_cnpb,unigram_s2_cnpb = 0,0,0,0,0,0

print("Q1 A) Subpart 4........................................................................................................")
trigrams_s1_cdp[PROB_dsc] = trigrams_s1_cdp[CNT_reconstd] / trigrams_s1_cdp[CNT_unsm]
trigrams_s1_cdp[PROB_dsc].replace(np.inf, 0, inplace=True)
trigrams_s1_cdp[PROB_dsc].replace(np.nan, 0, inplace=True)

# creating probability and count tables for bigrams for s1
bigram_s1_details = create_counts_prob(s1, 2)

# creating probability and count tables for unigrams for s1
unigram_s1_details = create_counts_prob(s1, 1)

temp = [Katz_backoff_prob(trigram, c, dp, 3, trigrams_s1_cdp[trigrams_s1_cdp[NGRAMS] == trigram], bigram_s1_details, unigram_s1_details, True) for
        trigram, c, dp in zip(trigrams_s1_cdp[NGRAMS], trigrams_s1_cdp[CNT_unsm], trigrams_s1_cdp[PROB_dsc])]
trigrams_s1_cdp[PROB_katz] = temp
print(f"For S1 ==> \n {trigrams_s1_cdp[[NGRAMS, PROB_katz]]}\n")
print(
    f"How many times you had to also compute the smoothed trigram probabilities (2 points) ==> {trigrams_s1_cnpb} \n")
print(f"How many times you had to compute the smoothed unigram probabilities (2 points) ==> {unigram_s1_cnpb} \n")

trigrams_s2_cdp[PROB_dsc] = trigrams_s2_cdp[CNT_reconstd] / trigrams_s2_cdp[CNT_unsm]
trigrams_s2_cdp[PROB_dsc].replace(np.inf, 0, inplace=True)
trigrams_s2_cdp[PROB_dsc].replace(np.nan, 0, inplace=True)

# creating probability and count tables for bigrams for s2
bigram_s2_details = create_counts_prob(s2, 2)

# creating probability and count tables for unigrams for s2
unigram_s2_details = create_counts_prob(s2, 1)

temp = [Katz_backoff_prob(trigram, c, dp, 3, trigrams_s2_cdp[trigrams_s2_cdp[NGRAMS] == trigram], bigram_s2_details, unigram_s2_details, False) for
        trigram, c, dp in zip(trigrams_s2_cdp[NGRAMS], trigrams_s2_cdp[CNT_unsm], trigrams_s2_cdp[PROB_dsc])]
trigrams_s2_cdp[PROB_katz] = temp
print("S2: \n ",trigrams_s2_cdp[[NGRAMS, PROB_katz]],"\n")
print(
    f"How many times you had to also compute the smoothed trigram probabilities (2 points) ==> {trigram_s2_cnpb} \n")
print(f"How many times you had to compute the smoothed unigram probabilities (2 points) ==> {unigram_s2_cnpb} \n")




#
# Compute the total probabilities for each sentence S1 and S2, when (a) using the trigram model without smoothing; (1 points) and
# (b) when using the trigram model Laplace-smoothed (1 points),
# as well when using the trigram probabilities resulting from the Katz back-off smoothing (1 points).
#

print("Q1 A) Subpart 5......................................................................................................\n")

print("(a) Total probabilities without smoothing (1 points) \n")
print("S1: ", trigrams_s1_cdp[[PROB_unsm]].sum().iloc[0],"\n")
print("S2: ", trigrams_s2_cdp[[PROB_unsm]].sum().iloc[0],"\n")
print("(b) Total probabilities with Laplace-smoothed (1 points)  \n")
print("S1: ", trigrams_s1_cdp[[PROB_sm]].sum().iloc[0],"\n")
print("S2: ", trigrams_s2_cdp[[PROB_sm]].sum().iloc[0],"\n")
print("(c) Total probabilities with Katz back-off smoothing (1 points) \n")
print("S1: ", trigrams_s1_cdp[[PROB_katz]].sum().iloc[0],"\n")
print("S2: ", trigrams_s2_cdp[[PROB_katz]].sum().iloc[0],"\n")

