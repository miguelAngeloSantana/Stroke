import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st = pd.read_csv("healthcare-dataset-stroke-data.csv")

st.head()
#st.info()

'''st = data.drop_duplicates(subset=None, keep='first', inplace=True) #REMOVENDO VALORES DUPLICADOS

#REMOVENDO VALORES MISSING
st = st.drop(st[st.bmi.isna()].index)

st.isnull().sum()'''


#ALTERANDO O TIPO DA VARIAVEL AGE DE FLOAT PARA INTEIRO
co_int = ['age']

def to_type(DataFrame, columns, type):
    for col in columns:
        DataFrame[col] = DataFrame[col].astype(type)

to_type(st, co_int, 'int')

st.head()

#PLOTANDO GRAFICOS COM AS VARIAVEIS DE VALOR UNICO
'''valores_unicos = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

st['smoking_status'].replace('Unknown', np.nan, inplace=True)

for f in valores_unicos:
    st[f].value_counts().plot(kind='bar')
    plt.title(f)
    plt.grid()
    plt.show()'''

#TRANSFORMANDO AS VARIAVEIS TARGETS EM VALORES ENTRE 0 E N_CLASES-1 
le = LabelEncoder()

en_st = st.apply(le.fit_transform)

en_st.head()

y = en_st['stroke']
x = en_st.drop('stroke', axis = 1)

X_treino, X_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

modelo = RandomForestClassifier()

modelo.fit(X_treino, y_treino)

pred = modelo.predict(X_teste)

pm = pd.Series(pred)

print(pm)