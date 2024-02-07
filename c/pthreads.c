#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>

const int THREAD_SIZE = 4;

float equations(int rank, float x, float y) {
    float functions[4];

    functions[0] = -0.2 * y * y * y + 2 * x;
    functions[1] = 0.3 * y;
    functions[2] = 0.01 * y * y - x * x;
    functions[3] = x / y;

  return functions[rank];
}

void runge_kutta(int rank, int n, float x0, float y0, float h, float y[]) {
	float k1, k2, k3, k4;
	y[0] = y0;

	for (int i = 1; i < n; i++) {
		k1 = h * equations(rank, x0, y[i - 1]);
		k2 = h * equations(rank, x0 + 0.5 * h, y[i - 1] + 0.5 * k1);
		k3 = h * equations(rank, x0 + 0.5 * h, y[i - 1] + 0.5 * k2);
		k4 = h * equations(rank, x0 + h, y[i - 1] + k3);

		y[i] = y[i - 1] + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);

		x0 = x0 + h;
	}
}

struct thread_args {
    int index;
    int n;
    float y0;
    float x0;
    float h;
    float *y;
};

void *thread_func(void *args){
    struct thread_args *st_args = (struct thread_args *)args;

    float *y = (float *)malloc(st_args->n * sizeof(float));

    runge_kutta(st_args->index, st_args->n, st_args->x0, st_args->y0, st_args->h, y);

    st_args->y = y;
    return NULL;
}

int main() {
    clock_t start_time = clock();

    pthread_t thread[THREAD_SIZE];
    struct thread_args arguments[THREAD_SIZE];

    float x0 = 0, x = 1, h = 0.1;
    float y0[] = {0.5, 15, 10, -20};
    int n = (int)((x - x0) / h);
    float y[n];

    for (int i = 0; i < THREAD_SIZE; i++){
        arguments[i].index = i;
        arguments[i].n = n;
        arguments[i].y0 = y0[i];
        arguments[i].x0 = x0;
        arguments[i].h = h;
        arguments[i].y = y;

        pthread_create(&thread[i], NULL, thread_func, &arguments[i]);
    }

    float gathered_y[THREAD_SIZE][n];

    for (int i = 0; i < THREAD_SIZE; i++){
        pthread_join(thread[i], NULL);

        for (int i = 0; i < THREAD_SIZE; i++) {
            for (int j = 0; j < n; j++) {
                gathered_y[i][j] = arguments[i].y[j];
            }
        }

        gathered_y[i] = arguments[i].y;
    }

    for (int i = 0; i < THREAD_SIZE; i++) {
        for (int j = 0; j < n; j++)
        {
            printf("%f ", gathered_y[i][j]);
        }
        printf("\n");
    }


    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("\nTotal execution time: %0.5f seconds\n", elapsed_time);
}