# импортирую библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# импоритрую датасет
df = pd.read_csv('male_players.csv')
#print(df.head()) # проверка

# отделяю игроков от вратарей
players = df[df['Position'] != 'GK']
#print(players.head(10)) # проверка

# сортирую только нужные статы
players = players[['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physicality', 'Overall']]
#print(players.head()) # проверка


# небольшой график для базового анализа
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(8,6), dpi=200)

axes[0][0].plot(players['Pace'],players['Overall'],'o', alpha=0.1)
axes[0][0].set_ylabel("Overall")
axes[0][0].set_xlabel("Pace")

axes[0][1].plot(players['Shooting'],players['Overall'],'o', alpha=0.1)
axes[0][1].set_ylabel("Overall")
axes[0][1].set_xlabel("Shooting")

axes[0][2].plot(players['Passing'],players['Overall'],'o', alpha=0.1)
axes[0][2].set_ylabel("Overall")
axes[0][2].set_xlabel("Passing")

axes[1][0].plot(players['Dribbling'],players['Overall'],'o', alpha=0.1)
axes[1][0].set_ylabel("Overall")
axes[1][0].set_xlabel("Dribbling")

axes[1][1].plot(players['Defending'],players['Overall'],'o', alpha=0.1)
axes[1][1].set_ylabel("Overall")
axes[1][1].set_xlabel("Defending")

axes[1][2].plot(players['Physicality'],players['Overall'],'o', alpha=0.1)
axes[1][2].set_ylabel("Overall")
axes[1][2].set_xlabel("Physicality")

plt.tight_layout()
#plt.show()

# общая зависимость между каждыми параметрами
sns.pairplot(players,diag_kind='kde')
#plt.show()

# разбиение на входные и выходные параметры
X = players.drop(['Overall'], axis=1)
y = players['Overall']
# проверка
#print(X.head())
#print(y.head())

# разбиение на тренировочный и проверочный датасет
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=72)

# создаю модель линейной регрессии
model = LinearRegression()
# обучение
model.fit(X_train, y_train)
# предсказание
test_predict = model.predict(X_test)
#print(test_predict[:5])

# mean: 66.52
#print(players['Overall'].mean())

#MAE: 2.88 ~ 4.3%
MAE = mean_absolute_error(y_test, test_predict)
#print(MAE)

#MSE: 12.76 ~ 19%, но это в квадрате
MSE = mean_squared_error(y_test, test_predict)
#print(MSE)

#RMSE: 3.57 ~ 5%
RMSE = np.sqrt(MSE)
#print(RMSE)

dump(model, 'models/lin_play_model.joblib')
# [0.03688373 0.06449819 0.20437359 0.15327724 0.09141388 0.26008192]
# ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physicality']
#print(model.coef_)

campaign = [[82, 75, 76, 82, 43, 48]]
#print(model.predict(campaign))


# создаю и обучаю модель с помощью полиномов

# Создать различные степени полинома
# Разбить данные на обучающие и тестовые наборы
# Обучить модель
# Сохранить метрики RMSE для обучаещего и тестового набора данных
# Нарисовать график с результатами - ошибка по степени полинома

train_rmse_errors =[]
test_rmse_errors = []

for d in range(1, 10):
    poly_converter = PolynomialFeatures(degree=d, include_bias=False)
    poly_features = poly_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, train_size=0.3, random_state=72)
    new_model = LinearRegression()
    new_model.fit(X_train, y_train)

    train_pred = new_model.predict(X_train)
    test_pred = new_model.predict(X_test)

    train_rmse = np.sqrt(mean_absolute_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_absolute_error(y_test, test_pred))

    train_rmse_errors.append(train_rmse)
    test_rmse_errors.append(test_rmse)

plt.figure(figsize=(12, 8))
plt.plot(range(1, 10), train_rmse_errors, label='TRAIN RMSE')
plt.plot(range(1, 10), test_rmse_errors, label='TEST RMSE')
plt.xlabel('Степень полинома')
plt.ylabel('RMSE')
plt.legend()
#plt.show()

final_poly_converter = PolynomialFeatures(degree=4, include_bias=False)
poly_model = LinearRegression()

full_converted_X = final_poly_converter.fit_transform(X)
poly_model.fit(full_converted_X, y)

transform = final_poly_converter.transform(campaign)
#print(poly_model.predict(transform))

trans = final_poly_converter.transform(X)
y_hat = poly_model.predict(trans)


fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(8,6), dpi=200)

axes[0][0].plot(players['Pace'],players['Overall'],'o', alpha=0.1)
axes[0][0].plot(players['Pace'],y_hat,'o', alpha=0.1)
axes[0][0].set_ylabel("Overall")
axes[0][0].set_xlabel("Pace")

axes[0][1].plot(players['Shooting'],players['Overall'],'o', alpha=0.1)
axes[0][1].plot(players['Shooting'],y_hat,'o', alpha=0.1)
axes[0][1].set_ylabel("Overall")
axes[0][1].set_xlabel("Shooting")

axes[0][2].plot(players['Passing'],players['Overall'],'o', alpha=0.1)
axes[0][2].plot(players['Passing'],y_hat,'o', alpha=0.1)
axes[0][2].set_ylabel("Overall")
axes[0][2].set_xlabel("Passing")

axes[1][0].plot(players['Dribbling'],players['Overall'],'o', alpha=0.1)
axes[1][0].plot(players['Dribbling'],y_hat,'o', alpha=0.1)
axes[1][0].set_ylabel("Overall")
axes[1][0].set_xlabel("Dribbling")

axes[1][1].plot(players['Defending'],players['Overall'],'o', alpha=0.1)
axes[1][1].plot(players['Defending'],y_hat,'o', alpha=0.1)
axes[1][1].set_ylabel("Overall")
axes[1][1].set_xlabel("Defending")

axes[1][2].plot(players['Physicality'],players['Overall'],'o', alpha=0.1)
axes[1][2].plot(players['Physicality'],y_hat,'o', alpha=0.1)
axes[1][2].set_ylabel("Overall")
axes[1][2].set_xlabel("Physicality")

plt.tight_layout()
#plt.show()

dump(final_poly_converter, 'models/final_converter.joblib')
dump(poly_model, 'models/poly_play_model.joblib')