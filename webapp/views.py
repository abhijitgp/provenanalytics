from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import json
import mpld3
from textblob import TextBlob
import re
from collections import Counter
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import gensim
from gensim import corpora, models
from webapp import app
from funclib import *

# Import data
filename = './webapp/serum_processed_u1subset.json'
ab = pd.read_json(filename,orient='index')
#Rename Acne category name.
ab.category = ab.category.replace('Acne &amp; Blemish Treatments', 'Acne')
top_category = ab.price.groupby(ab.category).sum().sort_values(ascending=False)
top_category = list(top_category.index[:4])
cd = ab[ab.category.isin(top_category)]

# Import LDA Model, GenSim Dictionary and Corpus
ldamodel = gensim.models.ldamodel.LdaModel.load('./webapp/lda0623.gz')
dictionary = dictionary = corpora.Dictionary.load('./webapp/dictionary0623.gz')
dctnry = corpora.Dictionary.load('./webapp/revs.dict')

# Import sc_reviews from JSON file, reset index and rename Acne category.
sc_reviews=pd.read_json('./webapp/sc_reviews.json')
sc_reviews = sc_reviews.reset_index(drop=True)
sc_reviews.category = sc_reviews.category.replace('Acne &amp; Blemish Treatments', 'Acne')

# Set plot style for Seaborn plots.
sns.set(style='ticks',palette='Set2')
sns.despine()

# app = Flask (__name__)
Bootstrap(app)

# Define the categories to be analyzed.
topcat = ['Acne','Face Masks','Face Moisturizer',\
          'Face Serums','Natural Skin Care','Sheet Masks']
          
# Define routes.
@app.route('/')
def index():
	return render_template ('index.html')

@app.route('/acne', methods=['POST','GET'])
def acne():
    ins = ent(topcat[0],sc_reviews,ab,cd, dctnry)
    return render_template ('analysis.html', ins = ins)

@app.route('/mask', methods=['POST','GET'])
def mask():
    ins = ent(topcat[1],sc_reviews,ab,cd, dctnry)
    return render_template ('analysis.html', ins = ins)

@app.route('/moisturizer', methods=['POST','GET'])
def moisturizer():
    ins = ent(topcat[2],sc_reviews,ab,cd, dctnry)
    return render_template ('analysis.html', ins = ins)

@app.route('/serum', methods=['POST','GET'])
def serum():
    ins = ent(topcat[3],sc_reviews,ab,cd, dctnry)
    return render_template ('analysis.html', ins = ins)

@app.route('/ncare', methods=['POST','GET'])
def ncare():
    ins = ent(topcat[4],sc_reviews,ab,cd, dctnry)
    return render_template ('analysis.html', ins = ins)

@app.route('/sheets', methods=['POST','GET'])
def sheets():
    ins = ent(topcat[5],sc_reviews,ab,cd, dctnry)
    return render_template ('analysis.html', ins = ins)

@app.route('/predict', methods=['POST','GET'])
def predict():
    return render_template ('predict.html')

# Define Prediction output route.
@app.route('/opred', methods=['POST','GET'])
def opred():
    if request.method == 'POST':
        newreview = request.form['newreview']
        vec = dictionary.doc2bow(str(newreview).split())
        tpks = ldamodel.get_document_topics(vec)
        final = str(round(TextBlob(newreview).sentiment.polarity,3))
        t = []
        w = []
        for i in tpks:
            t.append('Topic '+str(i[0]+1))
            w.append(round(i[1],3)) # Round to 3 decimals.
            
        tw = pd.DataFrame([t,w])
        tw = tw.transpose()
        tw.columns = ['Topics','Distribution']
        final2 = tw.to_html(index=False)
        rslt = {'tpks':final2,\
        'sentiment':final, 'newreview':newreview}
        return render_template ('opred.html', rslt = rslt)

if __name__=='__main__':
	app.run(debug=True)