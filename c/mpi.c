#include <stdio.h>
#include <mpi.h>

float equations(int rank, float x, float y) {
    float functions[4];

    functions[0] = -0.2 * y * y * y + 2 * x;
    functions[1] = 0.3 * y;
    functions[2] = 0.01 * y * y - x * x;
    functions[3] = x / y;

  return functions[rank];
}

void runge_kutta(int rank, int n, float x0, float y0, float x, float h, float y[])
{
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

double get_average(double array[]) {
    double sum = 0.0;
    int size = 10;

    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }

    return sum / size;
}

void test(int rank, int size, int n, float x0, float y0, float x, float h) {
        float y[n];
        double start_time = MPI_Wtime();
        runge_kutta(rank, n, x0, y0, x, h, y);

        float gathered_y[size][n];
        MPI_Gather(y, n, MPI_FLOAT, gathered_y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        double end_time;
        if (rank == 0) {
            double end_time = MPI_Wtime();

            printf("x: %f\n", x);
            printf("Time is: %0.8f\n", end_time - start_time);
        }
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float x0 = 0, x = 100, h = 0.1;
    float y0[] = {0.5, 15, 10, -20};
    int n = (int)((x - x0) / h) + 1;

        float y[n];
        double start_time = MPI_Wtime();
        runge_kutta(rank, n, x0, y0[rank], x, h, y);

        float gathered_y[size][n];
        MPI_Gather(y, n, MPI_FLOAT, gathered_y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

        double end_time;
        if (rank == 0) {
            double end_time = MPI_Wtime();

            printf("x: %f\n", x);
            printf("Time is: %0.8f\n", end_time - start_time);
                MPI_Finalize();
        }

	return 0;
}
