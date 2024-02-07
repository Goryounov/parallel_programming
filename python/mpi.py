from mpi4py import MPI
import numpy as np
import time

def equations(x, y):
    functions = [
        -0.2*y*y*y + 2*x,
        0.3*y,
        0.01*y*y - x**2,
        x / y
    ]
    return functions[rank]

def runge_kutta(f, y0, x0, xn, h):
    count = int((xn - x0) / h) + 1
    x_values = [x0]
    y_values = [y0]

    for i in range(1, count):
        x = x_values[i - 1]
        y = y_values[i - 1]

        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)

        y_new = y + (k1 + 2*k2 + 2*k3 + k4)/6
        x_new = x + h

        x_values.append(x_new)
        y_values.append(y_new)

    return np.array(y_values)

def test(xn):
    times = []
    if rank == 0:
        times = []

    for i in range(1, 10):
        start_time = 0
        if rank == 0:
            start_time = time.time()

        y = runge_kutta(equations, y0[rank], x0, xn, h)
        y = comm.gather(y, root=0)

        if rank == 0:
            y = np.vstack(y).T.tolist()

            end_time = time.time()
            times.append(end_time - start_time)

    average = 0
    if rank == 0:
        average = sum(times)/len(times)
    return average

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

y0 = [0.5, 15, 10, -20]
x0 = 0
xn = [1000, 5000, 10000, 25000, 50000]
h = 0.1

if rank == 0:
    print('=== Test result ===')
    print('Xn', 'Average time')

for i in range(0, len(xn)):
    average = test(xn[i])
    if rank == 0:
        print(xn[i], average)

if rank == 0:
    print('===================')