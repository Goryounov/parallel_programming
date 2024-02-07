import time

def equations(x, y):
  return [ -0.2 * y[0] * y[0] * y[0] + 2 * x,
           0.3 * y[1],
           0.01 * y[2] * y[2] - x**2,
           x / y[3]]

def runge_kutta(f, x0, xf , y0 , h):
  count = int((xf - x0) / h) + 1
  x, y = x0, [y0]

  for i in range(1, count):
    k1 = f(x, y[i-1])
    k2 = f(x + h / 2, [y + k * h / 2 for y, k in zip(y[i-1], k1)])
    k3 = f(x + h / 2, [y + k * h / 2 for y, k in zip(y[i-1], k2)])
    k4 = f(x + h, [y + k * h for y, k in zip(y[i-1], k3)])
    y.append([])

    for j in range(len(y0)):
      y[i]. append(y[i-1][j] + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]))

    x += h

  return y


def test(xn):
    times = []

    for i in range(1, 10):
        start_time = time.time()

        y = runge_kutta(equations, t0, xn, y0, 0.1)

        end_time = time.time()
        times.append(end_time - start_time)

    average = sum(times)/len(times)
    return average

y0 = [0.5, 15, 10, -20]
t0, tf = 0, 1000
xn = [1, 10, 100, 1000]

print('=== Test result ===')
print('Xn', 'Average time')
for i in range(0, 4):
    average = test(xn[i])
    print(xn[i], average)
print('===================')

