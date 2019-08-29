import pandas as pd
import re
from nltk import wordpunct_tokenize,pos_tag,sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, model_selection
import numpy as np
from nltk.corpus import wordnet as wn
from sklearn import  naive_bayes,metrics

df = pd.read_csv('raw1.csv')
df.head()

keyword = ['better than', 'compare to', 'higher than', 'but', 'challenge', 'prefer', 'beat', 'against']
s = []
for word in keyword:
    s.append(wn.synsets(word))
keywords=['better than','compare to','higher than','but','challenge','prefer','beat','against',
          'choose','competitor','iphone','htc','google pixel','pixel','huawei','samsung','oppo','vivo','nokia',
          'rivals','lenovo','xiaomi','redmi','mi','a2','most','outstanding','like','rival','best','superior','great','price range','the range'] #extend key list manually
tag=['JJR','RBR','JJS','RBS']# POS

support_set=0.0002# parameter supporter
confidence_set=0.5 #parameter confidence

#sentence tokenize,word tokenization,remove stopwords,pos
def preprocessing():
    stopword1 = ['i', ':', '.', '..', "'", ',', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
             "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
             'she', "she's", 'her', 'hers', 'herself','can',
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
             'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
             'a', 'an', 'the', 'and', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
             'about', 'into', 'through', 'during', 'before', 'after', 'from', 'up', 'down', 'in', 'on', 'off',
             'again', 'further', 'then', '...']

    df['words'] = None
    c = []
    v = 0
    for sen in df['comments']:
        sentence = sent_tokenize(str(sen))
        for senten in sentence:
            words=pos_tag(wordpunct_tokenize(senten))
            for word in words:
                m=word[0]
                if str(m).lower() not in stopword1:
                    c.append(word)
        df['words'][v]=c
        c=[]
        v=v+1
    a = 0
    df['ve'] = None #feature- just words
    df['pos']=None #feature-pos
    h=[]
    pos=[]
    for line in df['words']:
        for m in line:
            line1=m[0]
            ll=str(line1).replace("'", '').replace("[", '').replace("]", '')
            pattern = re.compile('[^a-zA-Z0-9_]')
            b = re.sub(pattern, ' ', ll)  # non-english and non-number
            h.append(b)
            pos.append(m[1])
        df['ve'][a] = h
        df['pos'][a]=pos
        h=[]
        pos=[]
        a = a + 1
    return(df['words'][0],df['pos'][0])

preprocessing()

def dfvepos_onlywords():
    p=0
    pp=0
    for line in df['ve']:
        a=str(line).replace("'", '').replace("[", '').replace("]", '').replace(',','')
        df['ve'][p]=a
        p=p+1
    for mm in df['pos']:
        mma=str(mm).replace("'", '').replace("[", '').replace("]", '').replace(',','')
        df['pos'][pp] =mma
        pp=pp+1
    return (df['ve'][0],df['pos'][0])

dfvepos_onlywords()

#######################POS and Keyword strategy#######
def PosKey_rule():#don't need to split the data as train or test data, it make prediction according to the rule, thus there has no overfitting problem.
    ps = 0
    df['key_predict'] = 0  # define all as noncomparative comments initially
    for line in df['ve']:
        for n in keywords:
            if n in str(line).lower():
                df['key_predict'][ps] = 1
        ps = ps + 1
    # tag to take into account
    sp = 0
    df['tag_predict'] = 0
    for line1 in df['pos']:
        for nn in tag:
            if nn in line1:
                df['tag_predict'][sp] = 1
        sp = sp + 1
    return (df['key_predict'][0], df['tag_predict'][0])
PosKey_rule()
#evaluation:keyword strategy
def keywords_evaluation():
    key_aa=metrics.f1_score(df['labels'],df['key_predict'],average='macro')
    key_rec=metrics.recall_score(df['labels'],df['key_predict'],average='macro')
    key_met=metrics.precision_score(df['labels'],df['key_predict'],average='macro')
    print("F-Score:",key_aa)# F-score: 0.6753706332798154
    print("recall:",key_rec)#same:0.7889711001176696
    print("precision:",key_met)#same:0.7104172387809525
keywords_evaluation()
#evaluation:POS strategy
def POS_evaluation():
    pos_aa=metrics.f1_score(df['labels'],df['tag_predict'])
    pos_rec=metrics.recall_score(df['labels'],df['tag_predict'])
    pos_met=metrics.precision_score(df['labels'],df['tag_predict'])
    print("F-Score:",pos_aa)# F-score: 0.7237715803452855
    print("recall:",pos_rec)#same:0.7962016070124178
    print("precision:",pos_met)#same:0.6634205721241632
POS_evaluation()

#######################CSR#############
def Generate_CSR_range():
    keywords=['better than','compare to','higher than','but','challenge','prefer','beat','against',
          'choose','competitor','iphone','htc','google pixel','pixel','huawei','samsung','oppo','vivo','nokia',
          'rivals','lenovo','xiaomi','redmi','mi','a2','most','outstanding','like','rival','best','superior','great','price range','the range']
    scale_size=3
    df['CSR_range']=None
    s0=[]
    f=0
    while f<= len(df['words']):
        for word,pos in df['words'][f]:
            if str(word).lower() in keywords:
                index=df['words'][f].index((word,pos))
                if len(df['words'][f])>1:
                    s0.append(df['words'][f][index-3:index+4])
                else:
                    pass
                df['words'][f].remove((word,pos))
        df['CSR_range'][f] =s0
        f=f+1
        s0 = []
    return (df['CSR_range'][0])
Generate_CSR_range()

def Generate_CSR_sequence():
    p1=0
    p2=[]
    p3=[]
    reset=0
    df['CSRseq']=None
    while reset<= len(df['words'])-1:
        for line in df['CSR_range'][reset]:
            for aaa,bbb in line:
                if str(aaa).lower() not in keywords:
                    p2.insert(p1,bbb)
                    p1=p1+1
                else:
                    p2.insert(p1,(str(aaa).lower()+'_'+bbb))
                    p1 = p1 + 1
            p3.append(p2)
            p2=[]
            p1=0
        df['CSRseq'][reset]=p3
        reset=reset+1
        p3=[]
Generate_CSR_sequence()

def Generate_CSRrule_andPredic():
    # CSR.csv
    rule = []  # 4701
    tag = []
    zz = 0
    for line in df['CSRseq']:
        for linea in line:
            if linea == []:
                pass
            else:
                tag.append((df['labels'][zz]))
                rule.append(linea)
        zz = zz + 1
    
    rulecount = []  # 4472
    for item in rule:
        if item not in rulecount:
            rulecount.append(item)
    
    # cover
    CSR0 = []
    counts = 0
    py = 0
    Y_N = 0
    for rul in rulecount:
        for rules in rule:
            if rul == rules:
                counts = counts + 1
                if tag[py] == 1:
                    Y_N = Y_N + 1
        support = Y_N / len(rule)
        confidence = Y_N / counts
        CSR0.append([rul, support, confidence])
        counts = 0
        Y_N = 0
        py = py + 1
    
    df['CSRv0'] = None
    nnn = 0
    for line in CSR0:
        df['CSRv0'][nnn] = line
        nnn = nnn + 1
    
    support_set = 0.0002  # 0.0002
    confidence_set = 0.5
    CSRrule = []  # 2425
    for rul, support, confidence in CSR0:
        if support > support_set and confidence > confidence_set:
            CSRrule.append((rul, support, confidence))
    
    # CSRprediction
    df['CSR_Pre'] = 0
    predictingposition = 0
    for line in df['CSRseq']:
        for rul, support, confidence in CSRrule:
            if rul in line:
                df['CSR_Pre'][predictingposition] = 1
        predictingposition = predictingposition + 1
Generate_CSRrule_andPredic()

def CSR_evaluation():
    key_aa = metrics.f1_score(df['labels'], df['CSR_Pre'], average='macro')
    key_rec = metrics.recall_score(df['labels'], df['CSR_Pre'], average='macro')
    key_met = metrics.precision_score(df['labels'], df['CSR_Pre'], average='macro')
    print("F-Score:", key_aa)  # F-score: 0.7006163515995589
    print("recall:", key_rec)  # 0.7039974504658244
    print("precision:", key_met)  # 0.6975875857875846
CSR_evaluation()
###################baseline experiment######
df.dropna(inplace=True)

vect = TfidfVectorizer(min_df=5).fit(df['ve'])  # TfidfVectorizer
word_vec = vect.transform(df['ve'])

# NB baseline experiment
def evaluate_naive():
    clfrNB=naive_bayes.MultinomialNB()
    vect = TfidfVectorizer(min_df=5).fit(df['ve'])  # TfidfVectorizer
    word_vec = vect.transform(df['ve'])
    predicted_labels = model_selection.cross_val_predict(clfrNB,word_vec,df['labels'], cv=5)  # pos,csr
    recalls = model_selection.cross_val_score(clfrNB, word_vec,df['labels'], cv=5, scoring='recall')
    precisions = model_selection.cross_val_score(clfrNB, word_vec,df['labels'], cv=5, scoring='precision')
    print('pricision：', np.mean(precisions), precisions)# 0.9194453284710541
    print('recall：', np.mean(recalls), recalls)#0.42663832517847117
evaluate_naive()
# SVM baseline experiment
def SVM_RAW(): #baseline experiment
    clfssvm = svm.SVC(kernel='linear', C=0.1)
    predicted_labels = model_selection.cross_val_predict(clfssvm,word_vec,df['labels'], cv=10)  # pos,csr
    recalls = model_selection.cross_val_score(clfssvm, word_vec,df['labels'], cv=5, scoring='recall')
    precisions = model_selection.cross_val_score(clfssvm, word_vec,df['labels'], cv=10, scoring='precision')
    print('pricision：', np.mean(precisions), precisions)#0.9982905982905983
    print('recall：', np.mean(recalls), recalls)#0.5134541857169594
SVM_RAW()



