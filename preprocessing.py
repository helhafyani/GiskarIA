import pandas as pd
import numpy as np
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer 

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")


class PreProcessing:
    
    def split_return_line(self,row):
        row_ = row[2]
        row_ = row_.replace('=20\n',' ')
        row_ = row_.replace('=\n','')
        row_ = row_.replace('=09','')
        tokens = [t for t in row[2].replace("\n\t", "").split('\n')] #split the text according to the '\n': back to line
        return tokens
    
    def clean_tokens(self,tokens):
        clean_tokens = tokens[:]
        for token in tokens:
            if token in ['',' ']: #or ':' not in token:
                clean_tokens.remove(token) #remove all the white space  
        return clean_tokens
    
    def create_dict(self,clean_tokens):
        
        email = {}
        message = ''
        keys_to_extract = ['date', 'subject']
        keys_not_extract = ['message-id', 'mime-version','content-type','content-transfer-encoding','x-from', \
                            'x-to','x-cc','x-bcc','x-folder','x-origin','x-filename','cc','email', 'sent','from', 'to','bcc']
        for line in clean_tokens:
            if all(x not in line for x in [':', 'Forwarded', 'Original Message']):
                message += line.strip()+str(' ')
                email['body'] = message

            elif all(x not in line for x in ['Forwarded', 'Original Message']): 
                pairs = line.split(':',1)
                key = pairs[0].lower()
                val = pairs[1].strip()
                if key in keys_to_extract:
                    if key not in email:
                        email[key] = val
                elif key not in keys_not_extract:
                    message += line.strip()+str(' ')
                    email['body'] = message
        return email
       
#         keys = [i.split(':',1)[0] for i in clean_tokens]

#         vals = []
#         for i in clean_tokens:
#             if len(i.split(':',1))<2:
#                 vals.append('tag_no_val')
#             elif len(i.split(':',1))==2:
#                 vals.append(i.split(':',1)[1])
        
#         pairs = list(zip(keys, vals))
#         dictionary = {}
#         for n, v in zip(keys, vals):
#             dictionary.setdefault(n, []).append(v)
        
#         return dictionary
    
#     def extract_Date(self,tokens_dictionary):
#         return tokens_dictionary['Date']
    
#     def extract_Subject(self,tokens_dictionary):
#         return tokens_dictionary['Subject'] 
    
#     def extract_Content(self,row):
#         content = row[2].split('Subject')[-1].split('\n',1)[-1]
#         content = content.split('\n')
#         content = self.clean_tokens(content)
#         content = ''.join(content)
#         return content
    
    def extratc_3features(self, row):
        tokens = self.split_return_line(row)
        clean_tokens = self.clean_tokens(tokens)
        tokens_dictionary = self.create_dict(clean_tokens)
        
        return tokens_dictionary
    
    def features_lists(self, df):
        Date = []
        Subject = []
        Content = []
        
        for index, row in df.iterrows():
    
            tokens_dictionary = self.extratc_3features(row)

            Date.append(tokens_dictionary['date'])
            Subject.append(tokens_dictionary['subject'])
            Content.append(tokens_dictionary['body'])
       
        df['Date'] = pd.to_datetime(Date)
        df['Subject'] = Subject
        df['Content'] = Content
        
        return df
    
class preprocess_subject:
    
    def clean_stopwords(self, row):
        clean_words = []
        words = word_tokenize(row)
        for w in words:
            if w not in stopwords.words('english'):
                clean_words.append(w)
        return clean_words
    
    def clean_punc(self, clean_words):
        words_no_punc = []
        for w in clean_words:
            if w.isalpha():
                words_no_punc.append(w.lower())
        return words_no_punc
    
    def words_to_remove(self, clean_words):
        
        texts=clean_words

        tokenList = []
        for i, sentence in enumerate(texts):
            doc = nlp(sentence)
            for token in doc:
                tokenList.append([i, token.text, token.lemma_, token.pos_, token.tag_, token.dep_])
        tokenDF = pd.DataFrame(tokenList, columns=["i", "text", "lemma", "POS", "tag", "dep"]).set_index("i")
        return tokenDF
    
    def clean_further_content(self, words_no_punc):
        clean_words = words_no_punc[:]
        tokenDF = self.words_to_remove(clean_words)
        words_to_remove = tokenDF[tokenDF.tag=='NNP'].text.values.tolist()
        for w in words_no_punc:
            if w in words_to_remove+['re', 'fw']:
                clean_words.remove(w)
        return clean_words
    
    def clean_further_subject(self, words_no_punc):
        clean_words = words_no_punc[:]
        for w in words_no_punc:
            if w in ['re', 'fw']:
                clean_words.remove(w)
        return clean_words
    
    def normalize(self, clean_words):
        
        lemme_words = [WordNetLemmatizer().lemmatize(w) for w in clean_words]
        return lemme_words
    
    def process_subject(self, row, function):
        clean_words = self.clean_stopwords(row)
        clean_words = function(clean_words)
        clean_words = self.clean_punc(clean_words)
        #clean_words = self.clean_further(clean_words)
        clean_words = self.normalize(clean_words)
        clean_words = " ".join(clean_words)
        return clean_words
    
    def clean_columns(self, df):
        df['clean_subject'] = df['Subject'].apply(lambda x: self.process_subject(x,self.clean_further_subject))
        df['clean_content'] = df['Content'].apply(lambda x: self.process_subject(x,self.clean_further_content))
#         clean_subject = []
#         clean_content = []
#         for index, row in df.iterrows():
#             clean_sub = self.process_subject(row[4])
#             clean_cont = self.process_subject(row[5])
#             clean_subject.append(clean_sub)
#             clean_content.append(clean_cont)
#         df['clean_subject'] = clean_subject
#         df['clean_content'] = clean_content
        return df