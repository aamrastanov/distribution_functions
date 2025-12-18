import numpy as np
import matplotlib.pyplot as plt

# параметры
k = 3
step = 1.0
sigma = k / 3.0
num_blocks = 6

def G(x, mu, sigma):
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    g[np.abs(x - mu) > 3 * sigma] = 0.0
    return g

# ось x
x = np.linspace(-5, step * k * (num_blocks + 1) + 5, 8000)

# центры mu_i
mu = [i * step for i in range(1, k * (num_blocks + 1) + 2)]

# F1(x)
f1 = np.zeros_like(x)
for b in range(num_blocks):
    block = np.ones_like(x)
    for m in range(k):
        block *= G(x, mu[b * k + m], sigma)
    f1 += (-1) ** b * block

# F2(x) — сдвиг на один элемент
f2 = np.zeros_like(x)
for b in range(num_blocks):
    block = np.ones_like(x)
    for m in range(k):
        block *= G(x, mu[b * k + m + 1], sigma)
    f2 += (-1) ** b * block

# F3(x) — сдвиг на два элемент
f3 = np.zeros_like(x)
for b in range(num_blocks):
    block = np.ones_like(x)
    for m in range(k):
        block *= G(x, mu[b * k + m + 2], sigma)
    f3 += (-1) ** b * block

# F4(x) — сдвиг на три элемент
f4 = np.zeros_like(x)
for b in range(num_blocks):
    block = np.ones_like(x)
    for m in range(k):
        block *= G(x, mu[b * k + m + 3], sigma)
    f4 += (-1) ** b * block

# F5(x) — сдвиг на четыре элемент
f5 = np.zeros_like(x)
for b in range(num_blocks):
    block = np.ones_like(x)
    for m in range(k):
        block *= G(x, mu[b * k + m + 4], sigma)
    f5 += (-1) ** b * block

# один график, разные цвета
plt.plot(x, f1, label="F1", color="blue")
plt.plot(x, f2, label="F2", color="red")
plt.plot(x, f3, label="F3", color="green")
plt.plot(x, f4, label="F4", color="yellow")
plt.plot(x, f5, label="F5", color="orange")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.legend()
plt.show()
