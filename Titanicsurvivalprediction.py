# ... existing imports ...
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv('Titanic-Dataset.csv')  


print(data.head())


print(data['Survived'].value_counts())  


sns.countplot(x='Survived', data=data)  
plt.title('Survival Distribution')
plt.show()



X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']] 
y = data['Survived']  


X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})  


X['Age'].fillna(X['Age'].median(), inplace=True)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()