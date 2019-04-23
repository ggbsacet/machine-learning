import pandas as pd
import sqlite3 

# The Amazon Fine Food Reviews dataset consists of reviews of fine foods from Amazon.<br>
# 
# Number of reviews: 568,454
# Number of users: 256,059
# Number of products: 74,258
# Timespan: Oct 1999 - Oct 2012
# Number of Attributes/Columns in data: 10 
# 
# Attribute Information:
# 
# 1. Id
# 2. ProductId - unique identifier for the product
# 3. UserId - unqiue identifier for the user
# 4. ProfileName
# 5. HelpfulnessNumerator - number of users who found the review helpful
# 6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
# 7. Score - rating between 1 and 5
# 8. Time - timestamp for the review
# 9. Summary - brief summary of the review
# 10. Text - text of the review
# 
# 
# #### Objective:
# Given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).

#Lets work on it
#create a connection to sqllite file/database
connection = sqlite3.connect('C:\\Gaurav Work\\ML\\machine-learning\\all_datasets_collection\\amazon-fine-food-reviews\\database.sqlite')

#pandas read_sql_query provides the ability to run a simple sql query and fetch data from database table
data = pd.read_sql_query("select * from reviews where score != 3", connection)
#here, we are ignoring the reviews having the rating 3 as we have consider these as neutral

#print the shape of data
print("The shape of data is - ", data.shape)

#prints the columns name
print(data.columns)

#print the count of each review rating
print("Count of each review ratings\n", data['Score'].value_counts())

#prints the first few rows to get the idea of data
print(data.head(5))

##Data Cleaning - Removing the duplicates and all
#checking how many duplicate records we have
dup_data = pd.read_sql_query("select userid, count(*) from reviews where score !=3 group by userid, score, summary, text, time having count(*) > 1", connection)
dup_data.shape
dup_data.head(5)

#Lets delete the dupliacte reviews from data as one user can not provide same text review at same time
data = data.drop_duplicates(subset={'UserId', 'Score', 'Summary','Text', 'Time'}, keep='first', inplace=False)
#print(sys.getsizeof(data))
print("The shape of data after removing duplication", data.shape)

#lets print the counts of each review rating after removing duplicates
print("Counts of each review ratings", data['Score'].value_counts())

##print("% removed for duplicates - ", )

#Lets convert Score from 1,2,4,5 to only two classes Positive (1) and Negative (0)
data['Score'] = data['Score'].apply(lambda x: 0 if x < 3 else 1)

#lets print the counts of positive and negative reviews
print("Counts of each review ratings", data['Score'].value_counts())


### TEXT PREPROCESSING
#Lets clear the text more
#remove html tags
#Remove special characters
#remove numbers to make sure no word is alpha-numeric
#make all lowercase
#Remove stopwords
#perform snowball stemming 

#print some random review text from data
sent_100 = data['Text'].values[100]
print(sent_100)
#print("="*50)      #it is not required.. only working as a seperator 

#Samples for Text preprocessing
mystring = "This isn't sample string to test alltext preprocessing. <html> tags are there. and we WOULD perform."

from bs4 import BeautifulSoup
mystring = BeautifulSoup(mystring, 'lxml').get_text()     #this will remove all html and xml tags

import re       #this is used to perform regular expression
mystring = re.sub(r'\S*\d\S*', '', mystring).strip()
mystring = re.sub('[^A-Za-z0-9]+', ' ', mystring)

def decontraction(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_all = set(stopwords.words('english'))
print("List of all stop words for english in nltk", stopwords_all)

stopwords_our = set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            
    'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
    "won't", 'wouldn', "wouldn't"])

mystring = ' '.join(s.lower() for s in mystring.split() if s.lower() not in stopwords_our)

processed_reviews = []

from tqdm import tqdm
for text in tqdm(data['Text'].values):
    text = BeautifulSoup(text, 'lxml').get_text()
    text = decontraction(text)
    text = re.sub(r'\S*\d\S*', ' ', text).strip()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = ' '.join(s.lower() for s in text.split() if s.lower() not in stopwords_our)
    processed_reviews.append(text.strip())

#lets print count of processed reviews
print("Total # of reviews processed ", len(processed_reviews))

#lets print a random processed review
print(processed_reviews[2354])

#lets save these processed reviews so we don;t need to perform same step again



#Lets create Bag of Words (BoW)
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
#count_vector.fit(processed_reviews)
#count_vector.transform(processed_reviews)

count_vector = count_vector.fit_transform(processed_reviews)
count_vector.get_shape()

#lets print count of each words in count vector 
#it would be a long list.. so avoid printing
print(count_vector.vocabulary_)

count_vector.get_feature_names()[:1000]

#lets do TSNE for BoW (with less reviews and features)
bow_reviews = proce
from sklearn.manifold import TSNE
model = TSNE(n_components=2, perplexity=5, n_iter=200)
tsne_data = model.fit_transform(count_vector[1000:].todense())


#lets create bi-gram 
bi_gram_count_vector = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)
bi_gram_count_vector.fit(processed_reviews)
final_counst_bi_gram = bi_gram_count_vector.fit_transform(processed_reviews)

#bi_gram_count_vector.vocabulary_

final_counst_bi_gram.get_shape()