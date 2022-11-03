import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# excel dosyasının csv halinin okunması
df = pd.read_csv('C:/Users/pc/vs_code/Flask3/iris.csv')

print(df.head())

# bağımsız ve bağımlı değişken seçin
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# veri kümesini train ve test olarak ayırın 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# modeli gerçekleştirelim
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)


classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))