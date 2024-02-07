#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
    double t;
    double *Y;
    double *YY;
    double *Y1;
    double *Y2;
    double *Y3;
    double *Y4;
    double *FY;
} RungeKutta;

void init(RungeKutta *rk, int N) {
    rk->Y = (double *)malloc(N * sizeof(double));
    rk->YY = (double *)malloc(N * sizeof(double));
    rk->Y1 = (double *)malloc(N * sizeof(double));
    rk->Y2 = (double *)malloc(N * sizeof(double));
    rk->Y3 = (double *)malloc(N * sizeof(double));
    rk->Y4 = (double *)malloc(N * sizeof(double));
    rk->FY = (double *)malloc(N * sizeof(double));
}

void set_init(RungeKutta *rk, double t0, double *Y0, int N) {
    rk->t = t0;
    for (int i = 0; i < N; i++) {
        rk->Y[i] = Y0[i];
    }
}

double *F(double t, double *Y, double *FY) {
    FY[0] = Y[0]*Y[0]*Y[0] + t;
    FY[1] = 0.3*Y[1];
    FY[2] = 0.01*Y[2]*Y[2] - t*t;
    FY[3] = t / Y[3];
    return FY;
}

void next_step(RungeKutta *rk, double dt, int N) {
    int i;

    if (dt < 0) return;

    F(rk->t, rk->Y, rk->Y1);

    for (i = 0; i < N; i++)
        rk->YY[i] = rk->Y[i] + rk->Y1[i] * (dt / 2.0);

    F(rk->t + dt / 2.0, rk->YY, rk->Y2);

    for (i = 0; i < N; i++)
        rk->YY[i] = rk->Y[i] + rk->Y2[i] * (dt / 2.0);

    F(rk->t + dt / 2.0, rk->YY, rk->Y3);

    for (i = 0; i < N; i++)
        rk->YY[i] = rk->Y[i] + rk->Y3[i] * dt;

    F(rk->t + dt, rk->YY, rk->Y4);

    for (i = 0; i < N; i++)
        rk->Y[i] = rk->Y[i] + dt / 6.0 * (rk->Y1[i] + 2.0 * rk->Y2[i] + 2.0 * rk->Y3[i] + rk->Y4[i]);

    rk->t = rk->t + dt;
}

double get_average(double array[]) {
    double sum = 0.0;
    int size = 10;

    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }

    return sum / size;
}

double test(int xn) {
    double times[10];
    double dt = 0.1;
    int N = 4;
    RungeKutta task;
    init(&task, N);

    for (int i = 0; i < 10; i++) {
        clock_t start_time = clock();
        double Y0[] = {0.5, 15, 10, -20};
        set_init(&task, 0, Y0, N);

        while (task.t <= xn) {
            //printf("x = %0.5f; %0.8f; %0.8f; %0.8f; %0.8f\n", task.t, task.Y[0], task.Y[1], task.Y[2], task.Y[3]);
            next_step(&task, dt, N);
        }

        clock_t end_time = clock();
        double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        times[i] = execution_time;
    }

    return get_average(times);
}

int main() {
    double average;
    int xn[10] = {100, 10, 1000, 2500, 5000, 7500, 10000, 50000, 100000, 200000};

    printf("=== Test result ===\n");
    printf("Xn Average time\n");

    for (int i = 0; i < 8; i++) {
        average = test(pow(10, i));
        printf("%f ", pow(10, i));
        printf("%0.8f\n", average);
    }

    printf("===================\n");

    return 0;
}