import pandas as pd
data=pd.read_csv('aggression_parsed_dataset.csv', engine='python', encoding='utf-8', error_bad_lines=False)

#libraries and modules needed
import pandas as pd
import re
import numpy as np
nltk.download('stopwords')
nltk.download('wordnet')
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

#functions needed
def get_part_of_speech_tags(token):
  """Maps POS tags to first character lemmatize() accepts.We are focusing on Verbs, Nouns, Adjectives and Adverbs here."""
  tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
  tag = nltk.pos_tag([token])[0][1][0].upper()
  return tag_dict.get(tag, wordnet.NOUN)
  
def preprocess(text):
  """Preprocesses text-deletes stopwords, lemmatizer, stemmer"""
  text=text.replace("<>,.()!@#$%^&*()_-+='`~';;[]{}\/", "")
  text=text.split()
  text_without_stopwords=[]
  #get rid of stopwords
  from nltk.corpus import stopwords
  stop = set(stopwords.words('english'))
  text_without_stopwords=[i for i in text if i not in stop and i.isalnum()]
  #lemmatize
  lemmatizer = WordNetLemmatizer()
  lemmatized_output_with_POS_information = [lemmatizer.lemmatize(token,get_part_of_speech_tags(token)) for token in text_without_stopwords]
  #stemmer
  stemmer2 = SnowballStemmer(language='english')
  output = [stemmer2.stem(token) for token in lemmatized_output_with_POS_information]
  return output
  
def create_vocabulary(l):
  vocabulary=[]
  for sentence in l:
    for word in preprocess(sentence):
      vocabulary.append(word)
  vocabulary=list(set(vocabulary))
  return vocabulary
 
def Bag_of_words(s, vocabulary):
  bag_of_words=[0]*(len(vocabulary))
  for i in range(len(vocabulary)):
    if vocabulary[i] in s:
      bag_of_words[i]+=1
  return bag_of_words
  
#preprocess data, create bag of words
txt=data["Text"].apply(preprocess)
vocabulary=create_vocabulary([i for i in data["Text"][:3000]])
#training and test set
training=txt[:3000].apply(lambda w: Bag_of_words(w,vocabulary))
training=pd.Series(training)
aggression_tag=pd.Series(data["oh_label"][:3000])
extracted_data=pd.DataFrame({"bag_of_word":training,"aggression":aggression_tag})

#shuffle data
from sklearn.utils import shuffle
shuffled = shuffle(extracted_data)
print(shuffled)

#test and training
training=extracted_data[:2600]
test=extracted_data[2600:]
train_x =list(training["bag_of_word"])
train_y =list(training["aggression"])
test_x=list(test["bag_of_word"])
test_y=list(test["aggression"])

#create machine learning model
#multiple-perceptron classifier
import sklearn
import numpy as np
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(train_x,train_y)

from sklearn.metrics import classification_report,confusion_matrix
predict_train = mlp.predict(train_x)
predict_test = mlp.predict(test_x)

#assess performance (macro avg=0.74)
print(confusion_matrix(test_y,predict_test))
print(classification_report(test_y,predict_test))

def aggression_detector():
  while True:
    string=input("Say something:")
    if mlp.predict([Bag_of_words_v2(string,vocabulary)])[0]==1:
      print("It seems that someone is aggressive. Watch your language, please :)")
    else:
      print("No aggression here, you're good to go. :)")
      
 aggression detector()
