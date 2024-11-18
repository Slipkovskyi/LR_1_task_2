import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

# Данні для XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # вхідні дані
y = np.array([0, 1, 1, 0])  # відповідні мітки (результат XOR)

# Створюємо модель двослойного персептрону
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='adam', max_iter=10000)

# Навчаємо модель
mlp.fit(X, y)

# Побудова розділяючої прямої
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Візуалізація
plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.coolwarm)
plt.title("Розділяюча пряма для XOR")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Виведемо коефіцієнти рівняння розділяючої прямої
# Візьмемо коефіцієнти для першого шару
weights = mlp.coefs_[0]
bias = mlp.intercepts_[0]

# Рівняння розділяючої прямої: w1*x1 + w2*x2 + b = 0
print(f"Рівняння розділяючої прямої: "
      f"{weights[0][0]:.2f}*x1 + {weights[1][0]:.2f}*x2 + {bias[0]:.2f} = 0"
      )
