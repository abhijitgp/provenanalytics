import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import mpld3
from collections import Counter
from gensim import corpora, models


def ent(cat,sc_reviews,ab,cd, dctnry):
    ''' This function plots various metrics for a specific category. Further, 
    using another fucntion, it also creates a list of top 10 words that 
    most frequently occurred in reviews with low or high star ratings.
    '''
    tmpmat = sc_reviews.sentiment[sc_reviews.category == cat]
    p = np.mean(tmpmat[tmpmat>=0])
    n = np.mean(tmpmat[tmpmat<0])
    data=[p, -n]
    myDF = pd.DataFrame(dict(data=data,label=['Positive','Negative']))
    
    # Figure 1
    img1 = plt.figure(figsize=(4,2))
    sns.set_style("whitegrid")
    sns.barplot(y=[p,-n], x=['Positive','Negative'],\
        linewidth=0)
    div = mpld3.fig_to_html(img1)
    
    # Figure 2    
    lsss = sc_reviews.sentiment[sc_reviews.category ==cat]
    img1 = plt.figure(figsize=(4,2))
    plt.hist(lsss, linewidth=0)
    plt.xlabel('Sentiment Polarity')
    plt.title ('Review Sentiment Histogram')
    f_one = mpld3.fig_to_html(img1)
    
    # Figure 3
    img2 = plt.figure(figsize=(4,2))
    sns.violinplot(x=cd.category,y=cd.price, \
        inner=None, orientation='vertical',linewidth=0)
    plt.title ('Market Segmentation')
    f_two = mpld3.fig_to_html(img2)
    
    # Figure 4
    topbrands = Counter(ab.brand[ab.category==cat]).most_common(5)
    tb = []
    for i in topbrands:
        tb.append(i[0])
    abtb= ab[ab.brand.isin(tb)]
    f_three = plt.figure(figsize=(4,2))
    sns.violinplot(x=abtb.brand,y=abtb.price,\
        inner=None, orientation='vertical',linewidth=0)
    f3 = mpld3.fig_to_html(f_three)
    
    # Figure 5
    abc = ab[ab.category==cat]
    f_four = plt.figure(figsize=(4,2))
    plt.scatter(abc.percent_recommend*10,abc.number_reviews,\
        color='r',lw=0)
    plt.xlabel('Likely to Recommend Product [%]')
    plt.ylabel('Number of Reviews')
    f4 = mpld3.fig_to_html(f_four)
    
    # Figure 6
    f_five = plt.figure(figsize=(4,2))
    plt.scatter(abc.review_rating,abc.number_reviews,lw=0)
    plt.xlabel('Review Ratings')
    plt.ylabel('Number of Reviews')
    f5 = mpld3.fig_to_html(f_five)

    # Get Most Common Words for Low and High-rated reviews.
    low_words_df = getTerms(cat,1, sc_reviews, dctnry)
    high_words_df = getTerms(cat,5, sc_reviews, dctnry)
    lw_df = low_words_df.T
    lw_df.columns = [' '.join('')]*lw_df.shape[1]
    lw_html = lw_df.to_html(index=False)
    hw_df = high_words_df.T
    hw_df.columns = [' '.join('')]*hw_df.shape[1]
    hw_html = hw_df.to_html(index=False)

    # Create a URL for Category-specific LDA Model.
    hurl = "./static/"+''.join(cat.split())+".html"
    
    # Pack all figures and other variables into a dictionary.
    ins = {'cat':cat,'dv':div,\
    'f1':f_one,'f2':f_two,'f3':f3, 'f4':f4,'f5':f5,\
    'low':lw_html,'high':hw_html,'hurl':hurl} #'sc':script,
    
    # Close all the plots for memory efficiency.
    plt.close('all')
    
    return ins


def getTerms(category,score, sc_reviews, dictionary):
    ''' Get top terms from a specific category and reviews belonging to 
    a specific rated reviews (such as 1-star-rated reviews.'''
    sub_corpus = sc_reviews.corpus[(sc_reviews.category==category) &\
     (sc_reviews.rating_score==score)]
    tfidf = models.TfidfModel(sub_corpus)
    corpus_tfidf = tfidf[sub_corpus]
    d = {dictionary.get(id): value for doc in corpus_tfidf for id, value in doc}
    dl = sorted(d.items(), key=lambda x: x[1], reverse=True)
    dldf = pd.DataFrame(dl[5:15],columns=['text', 'size'])
    return dldf