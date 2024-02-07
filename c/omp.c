#include <stdio.h>
#include <omp.h>

float equations(int rank, float x, float y) {
    float functions[4];

    functions[0] = -0.2 * y * y * y + 2 * x;
    functions[1] = 0.3 * y;
    functions[2] = 0.01 * y * y - x * x;
    functions[3] = x / y;

  return functions[rank];
}

void runge_kutta(int rank, int n, float x0, float y0, float x, float h, float y[]) {
    float k1, k2, k3, k4;

    y[0] = y0;
    for (int i = 1; i < n; i++)
    {
        k1 = h * equations(rank, x0, y[i - 1]);
        k2 = h * equations(rank, x0 + 0.5 * h, y[i - 1] + 0.5 * k1);
        k3 = h * equations(rank, x0 + 0.5 * h, y[i - 1] + 0.5 * k2);
        k4 = h * equations(rank, x0 + h, y[i - 1] + k3);

        y[i] = y[i - 1] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);

        x0 = x0 + h;
    }
}

double getAverage(double array[]) {
    double sum = 0.0;
    int size = 10;

    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }

    return sum / size;
}

int main(int argc, char *argv[]) {
    double start_time, end_time;
    float x0 = 0, x = 1000, h = 0.1;
    float y0[] = {0.5, 15, 10, -20};
    int n = (int)((x - x0) / h);
    int num_threads = 3;
    double times[10];

    for (int i = 0; i < 10; i++) {
       float result[num_threads][n];
       start_time = omp_get_wtime();

       #pragma omp parallel num_threads(num_threads)
       {
           int index = omp_get_thread_num();
           runge_kutta(index, n, x0, y0[index], x, h, result[index]);
       }

       end_time = omp_get_wtime();
       times[i] = end_time - start_time;
    }

    printf("Times\n");
    for (int i = 0; i < 10; i++) {
        printf("%.8f\n", times[i]);
    }

    printf("Average time: %.8f\n", getAverage(times));

    return 0;
}