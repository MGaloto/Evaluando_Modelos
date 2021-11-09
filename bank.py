import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from scipy import stats


df = pd.read_csv('bank-additional-full.csv',  sep = ';')


#%%

# Observamos la dimension del data set:
    
print('La dimension del data set es de', df.shape[0], 'filas y', df.shape[1], 'columnas')


#%%

# Observamos las primeras y ultimas 5 filas:

df.head(5)

df.tail(5)


#%%

# Observamos tipo de dato por columna:

df.dtypes


#%%

# Contar valores faltantes por columna


for column in df.isnull().columns.values.tolist():
    print('Variable: ',column)
    print('Total de NA: ',df.isna()[column].sum())
    print("")   



#%%


# Valores unicos para cada columna de caracteres

# Nos sirve para visualizar si alguno de los objetos tiene un valor unico 


for column in df.columns:
    if df[column].dtype == object:
        print(str(column), ':' , str(df[column].unique()))
        print(df[column].value_counts())
        print('__________________________')


#%%



# Observamos metricas de las variables del data set:

df.describe()

#%%

# Observamos la distribucion de cada una de las variables
# Para las cuantitativas un grafico de densidad y para las 
# cualitativas utilizamos grafico de barras




#%%

# Age
import matplotlib.pyplot as plt

sns.set_style("whitegrid")  
plt.figure(figsize = (10,5)) 
sns.distplot(x = df['age']  ,  bins = 20 , kde = True , color = 'red'
             , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable age", y=1.05, size=15)
plt.show()

#%%


# duration

sns.set_style("whitegrid")  
plt.figure(figsize = (10,5)) 
sns.distplot(x = df['duration']  ,  bins = 200 , kde = True , color = 'red'
             , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable age", y=1.05, size=15)
plt.show()


#%%


# campaign

plt.figure(figsize = (10,8))
sns.distplot(df['campaign'], color = 'red' , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable campaign", y=1.05, size=15)
plt.show()


#%%

# pdays

plt.figure(figsize = (10,8))
sns.distplot(df['pdays'], color = 'red' , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable pdays", y=1.05, size=15)
plt.show()



#%%


# previous


plt.figure(figsize = (10,8))
sns.distplot(df['previous'], color = 'red' , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable previous", y=1.05, size=15)
plt.show()


#%%


# emp.var.rate


plt.figure(figsize = (10,8))
sns.distplot(df['emp.var.rate'], color = 'red' , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable emp.var.rate", y=1.05, size=15)
plt.show()



#%%


# cons.price.idx


plt.figure(figsize = (10,8))
sns.distplot(df['cons.price.idx'], color = 'red' , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable cons.price.idx", y=1.05, size=15)
plt.show()



#%%


# cons.conf.idx


plt.figure(figsize = (10,8))
sns.distplot(df['cons.conf.idx'], color = 'red' , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable cons.conf.idx", y=1.05, size=15)
plt.show()



#%%



# euribor3m 


plt.figure(figsize = (10,8))
sns.distplot(df['euribor3m'], color = 'red' , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable euribor3m ", y=1.05, size=15)
plt.show()

#%%

#  nr.employed


plt.figure(figsize = (10,8))
sns.distplot(df['nr.employed'], color = 'red' , kde_kws=dict(linewidth = 2 , color = 'grey'))
plt.title("Distribucion de la variable nr.employed ", y=1.05, size=15)
plt.show()



#%%

#  y

ax = df['y'].value_counts().plot(kind='bar',
                                    figsize=(14,8),
                                    title="Frecuencia de los valores de Y",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()

#%%

#  poutcome

ax = df['poutcome'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de poutcome",
                                    color=plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()

#%%


#  day_of_week

ax = df['day_of_week'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de day_of_week",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()

#%%


#  month

ax = df['month'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de month",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()

#%%


# contact

ax = df['contact'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de contact",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()


#%%


# loan

ax = df['loan'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de loan",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()


#%%


# housing

ax = df['housing'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de housing",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()


#%%


# default

ax = df['default'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de default",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()


#%%



# education

ax = df['education'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de education",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()


#%%


# marital

ax = df['marital'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de marital",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()


#%%

# job 

ax = df['job'].value_counts().plot(kind='bar',
                                    figsize=(12,7),
                                    title="Frecuencia de los valores de job",
                                    color = plt.cm.Paired(np.arange(len(df))))
ax.set_xlabel("Categorias")
ax.set_ylabel("Frecuencia")
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(rotation=45)
plt.show()


#%%

# Observamos una matriz de correlacion entre las variable numericas

import matplotlib.pyplot as plt

corr = df.corr()
f, ax = plt.subplots(figsize=(10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap="PuRd", square=True, ax=ax, annot=True, linewidth=0.1)
plt.xticks(rotation=45)
plt.title("Coeficiente de Correlacion de Pearson", y=1.05, size=15)

# podemos visualizar una muy alta correlacion entre la variable euribor tasa a 3 meses y Tasa de variación del empleo. Tambien la euribor se correlaciona muy fuerte con el numero de empleados.
# El indice de precios del consumidor se correlaciona fuerte tambien con la tasa de variacion del empleo y eñ euribor


#%%

plt.subplots(figsize=(17,5))
sns.countplot(x = 'age', hue = 'y', data = df, palette = 'colorblind')



#%%


# Transformacion de etiquetas en columnas numericas:
    
from sklearn.preprocessing import LabelEncoder
    

for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column] = LabelEncoder().fit_transform(df[column])




#%%

# Dividimos los valores en X y en Y:
    
X = df.iloc[:, 0:df.shape[1]-1].values

Y = df.iloc[:, 20].values




#%%

# Creamos dos df para testear y para entrenar, para el testeo se usa el 25% del set de datos 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state=0)



#%%


# Modelo Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators= 10, criterion = 'entropy', random_state = 0) # la calidad de las divisiones va a ser por entropia, se estiman 10 arrboles de decision

random_forest.fit(X_train, Y_train)



#%%

# Precision del modelo con los datos entrenados:
    
print('El modelo tiene un ', random_forest.score(X_train, Y_train).round(4)*100, '% de precision en el conjunto de datos de entrenamiento')




#%%

# Matriz de confusion 


from sklearn.metrics import confusion_matrix

confusion_m = confusion_matrix(Y_test, random_forest.predict(X_test))




verdadero_negativo = confusion_m[0][0] # son las que se han clasificado bien de todo el testeo 

verdadero_positivo = confusion_m[1][1] # son las que se han clasificado bien de todo el testeo 

falso_negativo = confusion_m[1][0]

falso_positivo = confusion_m[0][1]


modelo_rand = ((verdadero_negativo + verdadero_positivo ) / ( verdadero_negativo + verdadero_positivo + falso_negativo + falso_positivo)).round(4)*100


# Luego se estima la precision del modelo testeando los valores que separamos en test:
    

print('El modelo tiene una precision derivada del testeo de: ' ,modelo_rand, '%' ) 


#%%

# Regresion logistica:
    
    
df_reg = pd.read_csv('bank-additional-full.csv',  sep = ';')

dummie_y = df_reg['y']

dummie_y= dummie_y.replace('yes', 1).replace('no', 0)


# Creamos el data set con variables dummies y luego agregamos la variable dependiente 

df_reg = pd.get_dummies(df_reg).iloc[:,0:-2]

df_reg = pd.concat([df_reg, dummie_y], axis=1)





#%%

# Creamos dos df para testear y para entrenar, para el testeo se usa el 25% del set de datos 


X = df_reg.iloc[:, 0:df_reg.shape[1]-1].values

Y = df_reg.iloc[:, 63].values


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25, random_state=0)





#%%


from sklearn.linear_model import LogisticRegression


log = LogisticRegression(max_iter = 6000)


log.fit(X_train, Y_train)



variables = list(df_reg.columns[:-1])

coeficientes = list(log.coef_[0])

datos = {'variables': variables, 'coeficientes': coeficientes}

dataframe = pd.DataFrame(datos)

#%%

from time import sleep

# Analizando los coeficientes con sus variables

for i in range(len(dataframe)):
    if dataframe['coeficientes'][i] < 0:
        print('Un incremento de la variable',dataframe['variables'][i], 'hacen caer las probabilidades de que el cliente haya suscripto un deposito a plazo dado que el coeficiente es: ',dataframe['coeficientes'][i].round(4) )
        sleep(5)
    else:
        print('Un incremento de la variable',dataframe['variables'][i], 'hacen aumentar las probabilidades de que el cliente haya suscripto un deposito a plazo dado que el coeficiente es: ',dataframe['coeficientes'][i].round(4) )
        sleep(5)
        
#%%

# Variables con un coeficiente relativamente mas elastico que la mayoria:
        
for i in range(len(dataframe)):
    if abs(dataframe['coeficientes'][i]) > 0.1:
        print('\nVariable: ',dataframe['variables'][i], '\nCoeficiente: ',dataframe['coeficientes'][i].round(4) )
        sleep(1)
        



#%%


from sklearn.metrics import confusion_matrix

y_pred = log.predict(X_test)
matriz = confusion_matrix(Y_test,y_pred)



#%%



verdadero_negativo_reg = matriz[0][0] # son las que se han clasificado bien de todo el testeo 

verdadero_positivo_reg = matriz[1][1] # son las que se han clasificado bien de todo el testeo 

falso_negativo_reg = matriz[1][0]

falso_positivo_reg = matriz[0][1]


modelo_reg = (((verdadero_negativo_reg + verdadero_positivo_reg ) / ( verdadero_negativo_reg + verdadero_positivo_reg + falso_negativo_reg + falso_positivo_reg))*100).round(2)


# Luego se estima la precision del modelo testeando los valores que separamos en test:
    

print('El modelo tiene una precision derivada del testeo de: ' ,modelo_reg, '%' ) 



#%%


# Conclusion de los modelos


if modelo_reg > modelo_rand:
    print('El modelo con mayor precision es el modelo de Regresion Logistica con un ', modelo_reg, '% de precision sobre el test')
else:
    print('El modelo con mayor precision es el modelo de Random Forest con un ', modelo_rand, '% de precision sobre el test')





#%%




