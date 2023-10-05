from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузим набор данных Iris для задачи классификации цветков ириса
iris = load_iris()
X = iris.data  # Признаки
y = iris.target  # Метки классов

# Разделим данные на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создадим модель k-ближайших соседей с числом соседей равным 3 и использованием Евклидова расстояния
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# Обучим модель на тренировочных данных
knn.fit(X_train, y_train)

# Сделаем прогноз на тестовых данных
y_pred = knn.predict(X_test)

# Оценим точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')
