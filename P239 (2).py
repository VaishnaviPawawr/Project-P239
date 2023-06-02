#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
df = pd.read_csv("D:/EXCELR/Project/tweet.csv")






cleaned_df = df



# In[58]:


cleaned_df = pd.DataFrame(cleaned_df)


# In[59]:


#nltk.download('punkt')
import nltk
from nltk.tokenize import word_tokenize

# Tokenize the tweets
cleaned_df['tweets'] = cleaned_df['tweets'].apply(nltk.word_tokenize)



# In[60]:


cleaned_df['tweets'] = cleaned_df['tweets'].astype(str)


# In[61]:


#remove special characters and punctuation from the 'tweets' column
import re

# Define a function to remove special characters and punctuation
def remove_special_chars(text):
    # Remove any URLs
    text = re.sub(r'http\S+', '', text)
    # Remove any non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove any digits
    text = re.sub(r'\d+', '', text)
     # Remove any mentions
    text = re.sub(r'@\w+', '', text)
    return text


# In[62]:


# Apply the function to the 'tweets' column
cleaned_df['tweets'] = cleaned_df['tweets'].apply(remove_special_chars)



# In[63]:


import nltk
#nltk.download('stopwords')  # download stopwords if not already downloaded
from nltk.corpus import stopwords


# In[64]:



# Define the stopwords
stop_words = set(stopwords.words('english'))

# Apply stopwords removal using a lambda function
cleaned_df['tweets'] = cleaned_df['tweets'].apply(lambda x: ' '.join([word for word in x.lower().split() if word not in stop_words]))



# In[ ]:





# In[65]:





# In[66]:


from nltk.stem import WordNetLemmatizer

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Apply lemmatization to the 'tweets' column in the cleaned_df DataFrame
cleaned_df['tweets'] = cleaned_df['tweets'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))



# In[67]:


###EDA Starts :-

## Number of tweets
#num_tweets = len(cleaned_df['tweets'])

# Average tweet length
#avg_tweet_length = cleaned_df['tweets'].apply(len).mean()

# Maximum tweet length
#max_tweet_length = cleaned_df['tweets'].apply(len).max()

# Minimum tweet length
#min_tweet_length = cleaned_df['tweets'].apply(len).min()

#print("Number of tweets:", num_tweets)
#print("Average tweet length:", avg_tweet_length)
#print("Maximum tweet length:", max_tweet_length)
#print("Minimum tweet length:", min_tweet_length)


# In[68]:


#import matplotlib.pyplot as plt

# Count the number of tweets in each class
#class_counts = cleaned_df['class'].value_counts()
#print(class_counts)
# Plotting a bar chart
#class_counts.plot(kind='bar', rot=0)
#plt.xlabel('Class')
#plt.ylabel('Number of Tweets')
#plt.title('Class Distribution')
#plt.show()

# Plotting a pie chart
#class_counts.plot(kind='pie', autopct='%1.1f%%')
#plt.title('Class Distribution')
#plt.show()


# In[69]:


#from collections import Counter
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt

# Combine all tweets into a single string
#all_tweets = ' '.join(cleaned_df['tweets'])

# Split the string into individual words
#words = all_tweets.split()

# Count the frequency of each word
#word_freq = Counter(words)

# Get the most common words and their frequencies
#top_words = word_freq.most_common(10)

# Create a bar chart of the top words
#x = [word[0] for word in top_words]
#y = [count[1] for count in top_words]

#plt.bar(x, y)
#plt.xlabel('Words')
#plt.ylabel('Frequency')
#plt.title('Top 10 Most Frequent Words')
#plt.xticks(rotation=45)
#plt.show()

# Create a word cloud of all words
#wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tweets)

#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#plt.title('Word Cloud of All Words')
#plt.show()


# In[70]:


#from nltk.sentiment import SentimentIntensityAnalyzer
#import matplotlib.pyplot as plt

# Initialize the SentimentIntensityAnalyzer
#sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each tweet and assign sentiment labels
#sentiments = cleaned_df['tweets'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Categorize the sentiment labels as positive, negative, or neutral
#sentiment_labels = ['Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral' for score in sentiments]

# Count the number of tweets in each sentiment category
#sentiment_counts = pd.Series(sentiment_labels).value_counts()

# Plotting a pie chart
#sentiment_counts.plot(kind='pie', autopct='%1.1f%%')
#plt.title('Sentiment Distribution')
#plt.show()

# Plotting a histogram
#plt.hist(sentiments, bins=20, edgecolor='k')
#plt.xlabel('Sentiment Score')
#plt.ylabel('Number of Tweets')
#plt.title('Sentiment Distribution')
#plt.show()"""


# In[71]:


# Separating Figurative classified data
tweets_figurative=cleaned_df[cleaned_df['class']=='figurative']
#tweets_figurative
import pandas as pd 
# Create a dataframe from the list of words
df1 = pd.DataFrame(tweets_figurative)
df1 = df1["tweets"].tolist()
#df1


# In[ ]:





# In[72]:


# Separating Irony classified data
tweets_irony=cleaned_df[cleaned_df['class']=='irony']
df2 = pd.DataFrame(tweets_irony)
df2 = df2["tweets"].tolist()
#df2


# In[73]:


# Separating Sarcasm classified data
tweets_sarcasm=cleaned_df[cleaned_df['class']=='sarcasm']
df3 = pd.DataFrame(tweets_sarcasm)
df3 = df3["tweets"].tolist()
#df3


# In[74]:


# Separating Regular classified data
tweets_regular=cleaned_df[cleaned_df['class']=='regular']
df4 = pd.DataFrame(tweets_regular)
df4 = df4["tweets"].tolist()
#df4


# In[75]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Concatenate the lists into a single list
Combined_list = df4 + df3 + df1 + df2

# Create a list of labels corresponding to the combined_list
labels = ['Regular_list'] * len(df4) + ['sarcasm'] * len(df3) + ['irony'] * len(df2) + ['figurative'] * len(df1)


# In[76]:


# Split the combined_list and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Combined_list, labels, test_size=0.2, random_state=42)


# In[77]:


# Create a TF-IDF vectorizer with n-gram range (1, 3) and maximum of 6000 features
vectorizer1 = TfidfVectorizer(ngram_range=(1, 3), max_features=300)

# Fit the vectorizer on the training data and transform the tweets into TF-IDF features
X_train_tfidf = vectorizer1.fit_transform(X_train)

# Transform the testing tweets into TF-IDF features
#X_test_tfidf = vectorizer1.transform(X_test)


# In[78]:


"""# Get the feature names
feature_names = vectorizer.get_feature_names()

# Print the feature names
for feature_name in feature_names:
    print(feature_name)


# In[ ]:





# In[ ]:





# In[ ]:"""





# In[81]:
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

# Train an SVM classifier
classifier1 = SVC(kernel='linear', random_state=42)
classifier1.fit(X_train_tfidf, y_train)

# Predict the class labels for the testing tweets
#y_pred = classifier1.predict(X_test_tfidf)

# Compute accuracy score
#accuracy = accuracy_score(y_test, y_pred)

# Compute precision score
#precision = precision_score(y_test, y_pred, average='weighted')

# Compute recall score
#recall = recall_score(y_test, y_pred, average='weighted')

# Compute F1 score
#f1 = f1_score(y_test, y_pred, average='weighted')"""

# Print the scores
#print("Accuracy:", accuracy)
#print("Precision:", precision)
#print("Recall:", recall)
#print("F1 Score:", f1)"""


# In[82]:

#from sklearn.metrics import classification_report
# Evaluate the performance of the classifier
#print(classification_report(y_test, y_pred))


# In[ ]:
