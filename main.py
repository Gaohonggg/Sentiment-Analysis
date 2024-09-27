import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re
import string
import nltk
#nltk.dowload("stopwords")
#nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("IMDB-Dataset.csv")
a = len(df)
df = df.drop_duplicates() # remove duplicate rows

stop = set(stopwords.words("english")) # Các từ không có nhiều ý nghĩa ảnh hưởng trong english

def expand_contractions(text):
    return contractions.fix(text) # Chuyển các từ viết tắt thành đầy đủ

def preprocess_text(text):
    w1 = WordNetLemmatizer()  # Chuyển đổi thành các từ dạng gốc
    soup = BeautifulSoup(text,"html.parser")

    text = soup.get_text()
    text = expand_contractions(text)

    emoji_clean = re.compile("["
                             u"\U0001F600-\U0001F64F"  
                             u"\U0001F300-\U0001F5FF"  
                             u"\U0001F680-\U0001F6FF"  
                             u"\U0001F1E0-\U0001F1FF"  
                             u"\U00002702-\U000027B0"
                             u"\U000024C2-\U0001F251"
                             "]+", flags = re.UNICODE )

    text = emoji_clean.sub(r'',text)
    text = re.sub(r'\.(?=\S)','. ',text)   #Tìm dựa trên biểu thức chính quy và thay thế tương ứng.
    text = re.sub(r'http\S+', '', text)
    text = "".join([
        word.lower() for word in text if word not in string.punctuation
    ])
    text = " ".join([
        w1.lemmatize(word) for word in text.split() if word not in stop and word.isalpha()
    ])
    return text

df["review"] = df["review"].apply( preprocess_text )

def func(pct, allvalues):
    absolute = int( pct / 100.*np.sum(allvalues) )
    return "{:.1f}%\n({:d})".format(pct,absolute)

freq_pos = len(df[df["sentiment"]=="positive"])
freq_neg = len(df[df["sentiment"]=="negative"])

data = [freq_pos,freq_neg]
labels = ["positive","negative"]

pie, ax = plt.subplots(figsize=[11,7])
plt.pie(x=data,
        autopct=lambda pct: func(pct,data),
        explode=[0.0025]*2,
        pctdistance=0.5,
        colors=[sns.color_palette()[0],"tab:red"],
        textprops={"fontsize":16})
plt.legend([r'Positive',r'Negative'],loc="best",prop={"size":14})
pie.savefig("PieChart.png")
plt.show()


words_len = df["review"].str.split().map(lambda x: len(x))
df_temp = df.copy()
df_temp["words length"] = words_len

hist_positive = sns.displot(
    data=df_temp[ df_temp["sentiment"]=="positive" ],
    x="words length",
    hue="sentiment",
    kde=True,
    height=7,
    aspect=1.1,
    legend=False
).set(title="Words in positive reviews")
plt.show()

hist_negative = sns.displot(
    data=df_temp[ df_temp["sentiment"]=="negative" ],
    x="words length",
    hue="sentiment",
    kde=True,
    height=7,
    aspect=1.1,
    legend=False
).set(title="Words in negative reviews")
plt.show()

plt.figure(figsize=(7,7.1))
kernel_distibution_number_words_plot = sns.kdeplot(
    data=df_temp,
    x="words length",
    hue="sentiment",
    fill=True,
    palette=[sns.color_palette()[0],"red"]
).set(title="Words in reviews")
plt.legend(title="Sentiment",labels=["negative","positive"])
plt.show()


label_encode = LabelEncoder()
y_data = label_encode.fit_transform(df["sentiment"])

X_train, X_test, y_train, y_test = train_test_split(df["review"],y_data
                                                    ,test_size=0.2,
                                                    random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=100000)
tfidf_vectorizer.fit(X_train)

X_train_encoded = tfidf_vectorizer.transform(X_train)
X_test_encoded = tfidf_vectorizer.transform(X_test)

dt_classifier = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)
dt_classifier.fit(X_train_encoded,y_train)
y_pred = dt_classifier.predict(X_test_encoded)
print( accuracy_score(y_pred,y_test) )

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_encoded,y_train)
y_pred = rf_classifier.predict(X_test_encoded)
print( accuracy_score(y_pred,y_test) )



tc = "This movie was absolutely fantastic, I loved it!"
tc = preprocess_text(tc)
tc = tfidf_vectorizer.transform([tc])

dt_pred = dt_classifier.predict(tc)
rf_pred = rf_classifier.predict(tc)

prediction_label_dt = label_encode.inverse_transform(dt_pred)
prediction_label_rf = label_encode.inverse_transform(rf_pred)

print("Decision Tree Prediction:", prediction_label_dt[0])
print("Random Forest Prediction:", prediction_label_rf[0])

tc = "This movie was absolutely fantastic,but I hated it!"
tc = preprocess_text(tc)
tc = tfidf_vectorizer.transform([tc])

dt_pred = dt_classifier.predict(tc)
rf_pred = rf_classifier.predict(tc)

prediction_label_dt = label_encode.inverse_transform(dt_pred)
prediction_label_rf = label_encode.inverse_transform(rf_pred)

print("Decision Tree Prediction:", prediction_label_dt[0])
print("Random Forest Prediction:", prediction_label_rf[0])


































