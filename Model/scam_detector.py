import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE 
import joblib
from imblearn.over_sampling import RandomOverSampler


df = pd.read_csv('Model/SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])

print(df['label'].value_counts())

df = df.dropna(subset=['label', 'message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna(subset=['label'])

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop('label', axis=1)  
y = df['label']  

oversampler = RandomOverSampler()

X_resampled, y_resampled = oversampler.fit_resample(X, y)

df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_balanced['label'] = y_resampled

df_balanced['label'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train.fillna(''))
X_test_tfidf = vectorizer.transform(X_test.fillna(''))

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

model = LogisticRegression()
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, 'Model/scam_model.pkl')
joblib.dump(vectorizer, 'Model/vectorizer.pkl')

print(df['label'].value_counts())
