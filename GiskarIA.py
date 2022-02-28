#!/usr/bin/env python
# coding: utf-8

# In[42]:


from utils import *


# In[4]:


df = pd.read_csv("giskard_dataset.csv", sep=";")


# In[5]:


classes = {'internal company policy':7, 'alliances / partnerships':9,
       'internal company operations':8,
       'internal projects -- progress and strategy':2,
       'regulations and regulators (includes price caps)':1,
       ' company image -- current':3,
       'california energy crisis / california politics':6,
       'meeting minutes':12,
       'political influence / contributions / contacts':5, 'legal advice':10,
       'talking points':11, 'company image -- changing / influencing':4,
       'trip reports':13}


# In[6]:


df.Target = df.Target.map(classes)
df.head()


# In[7]:


from test import PreProcessing, preprocess_subject


# In[8]:


PP = PreProcessing()
df = PP.features_lists(df)


# In[9]:


PS = preprocess_subject()
df = PS.clean_columns(df)


# In[10]:


df.head()


# In[43]:


df.head()


# In[11]:


df.to_csv('data_v4.csv', index=False)


# # Mine the date column

# In[12]:


weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
df['days'] = df['Date'].apply(lambda x: calendar.day_name[x.weekday()])
df['weekdays'] = ['weekday' if item in weekdays else 'weekend' for item in df['days']]
df['month'] = df['Date'].apply(lambda x: calendar.month_name[x.month])
df['office_hours'] = df['Date'].apply(lambda x: x.hour)
df['office_hours'] = ['midday' if item >=8 and item <= 18 else 'midnight' for item in df['office_hours']]
df['year'] = df['Date'].apply(lambda x: x.year)
df['year'] =df['year'].apply('str')
df['new_date'] = df['office_hours'] + str(' ') + df['weekdays'] + str(' ') + df['weekdays'] + str(' ') + df['month'] + str(' ') + df['year']


# In[13]:


df['test'] = df.clean_subject + str(' ')+df.clean_content+ str(' ')+df.new_date


# # Classification

# In[46]:


X = df[['clean_subject','clean_content','test']]
y = df['Target']


# In[47]:


#all
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC())
    ])

pipeline.fit(X_train['test'], y_train)

# Predit using the trained model
y_pred_test = pipeline.predict(X_test['test'])

print(accuracy_score(y_pred_test, y_test))


# # Imbalanced data: SMOTE

# In[48]:


from imblearn.over_sampling import SMOTE
import seaborn as sns


# In[49]:


x=df['Target'].value_counts()
print(x)
sns.barplot(x.index,x)


# In[50]:


X = df[['clean_subject','clean_content','test']]
y = df['Target']


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# # Une seule colonne qui contient toutes les informations

# In[52]:


vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(X['test'])

tfidfconverter = TfidfTransformer()  
X = tfidfconverter.fit_transform(X)


smote = SMOTE('not majority',k_neighbors=2,random_state=40)

X_sm, y_sm = smote.fit_resample(X, y)
print(X_sm.shape, y_sm.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_sm, y_sm, test_size=0.25, random_state=42
)

clf = SVC().fit(X_train , y_train)

y_pred = clf.predict(X_test) 

print(accuracy_score(y_pred, y_test))


# # Multi-view learning

# In[53]:


X= df[['clean_subject','clean_content','test', 'new_date']]
y = df['Target']


# In[54]:


vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X_1 = vectorizer.fit_transform(X['clean_content'])
X_2 = vectorizer.fit_transform(X['clean_subject'])
X_3 = vectorizer.fit_transform(X['new_date'])

tfidfconverter = TfidfTransformer()  
X_1 = tfidfconverter.fit_transform(X_1)
X_2 = tfidfconverter.fit_transform(X_2)
X_3 = tfidfconverter.fit_transform(X_3)


# In[55]:


print('X_1: ', X_1.shape)
print('X_2: ', X_2.shape)
print('X_3: ', X_3.shape)


# In[56]:


from scipy.sparse import hstack
X_ = hstack((X_1, X_2, X_3))


# In[57]:


smote = SMOTE('not majority',k_neighbors=2,random_state=40)

X_sm, y_sm = smote.fit_resample(X_, y)
print(X_sm.shape, y_sm.shape)


# In[58]:


y_s=y_sm.value_counts()
print(y_s)
sns.barplot(y_s.index,y_s)


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(
    X_sm, y_sm, test_size=0.25, random_state=42
)


# In[61]:


#subject
X2 = X_train.tocsr()[:,1500:1638]

clf = SVC().fit(X2 , y_train)

X2_test = X_test.tocsr()[:,1500:1638]
y_pred_sub = clf.predict(X2_test) 

print(accuracy_score(y_pred_sub, y_test))


# In[62]:


#content
X1 = X_train.tocsr()[:,:1500]

clf = SVC().fit(X1 , y_train)

X1_test = X_test.tocsr()[:,:1500]
y_pred_cont = clf.predict(X1_test) 

print(accuracy_score(y_pred_cont, y_test))


# In[63]:


#date
X3 = X_train.tocsr()[:,1638:]

clf = SVC().fit(X3 , y_train)

X3_test = X_test.tocsr()[:,1638:]
y_pred_date = clf.predict(X3_test) 

print(accuracy_score(y_pred_date, y_test))


# In[64]:


df_meta = pd.DataFrame({'subject':y_pred_sub, 'content':y_pred_cont, 'date':y_pred_date, 'target':y_test})


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(df_meta[['subject', 'content','date']], df_meta['target'], test_size=0.33, random_state=42)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train) 

y_pred = classifier.predict(X_test) 

print(accuracy_score(y_pred, y_test))


# In[ ]:




