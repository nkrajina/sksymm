#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>
#include <omp.h>
#include "pdk.h"

#define error(str, ...)		{											\
								fprintf(stderr, str, __VA_ARGS__);		\
								exit(EXIT_FAILURE);						\
							}

int main(int argc, char **argv)
{
	if (argc != 6)
		error("usage: %s m n A.dat G.dat V.dat\n", argv[0]);

	int m = atoi(argv[1]);
	int n = atoi(argv[2]);

	if (m <= 0 || n <= 0 || n % 2)
		error("%s: matrix dimensions must be positive and n must be even\n", argv[0]);

	// opening files
	FILE *fA = fopen(argv[3], "r");

	if (fA == NULL)
		error("%s: can't open file %s\n", argv[0], argv[3]);

	FILE *fG = fopen(argv[4], "w");

	if (fG == NULL)
		error("%s: can't open file %s\n", argv[0], argv[4]);

	FILE *fV = fopen(argv[5], "w");

	if (fV == NULL)
		error("%s: can't open file %s\n", argv[0], argv[5]);

	// initializing matrix A
	int lda = 2 * VECLEN * ((m + 2 * VECLEN - 1) / (2 * VECLEN));
	double *A = (double *) aligned_alloc(VECLEN * sizeof(double), lda * n * sizeof(double));

	if (A == NULL)
		error("%s: error allocating memory\n", argv[0]);

	dlaset("n", &lda, &n, &fp64_zero, &fp64_zero, A, &lda);

	for (int j = 0; j < n; ++j)
		if (fread(A + j * lda, sizeof(double), m, fA) != m)
			error("%s: can't read data from file %s\n", argv[0], argv[3]);

	fclose(fA);

	// initializing matrix V
	int ldv = 2 * VECLEN * ((n + (2 * VECLEN) - 1) / (2 * VECLEN));
	double *V = (double *) aligned_alloc(VECLEN * sizeof(double), ldv * n * sizeof(double));

	if (V == NULL)
		error("%s: error allocating memory\n", argv[0]);

	dlaset("n", &ldv, &n, &fp64_zero, &fp64_one, V, &ldv);

	// allocating extra storage for rotation counting
	int *nrots = (int *) aligned_alloc(VECLEN * sizeof(int), omp_get_max_threads() * sizeof(int));

	if (nrots == NULL)
		error("%s: error allocating memory\n", argv[0]);


	struct timespec start_time, finish_time;
	clock_gettime(CLOCK_REALTIME, &start_time);

	pdk(&m, &n, A, &lda, V, &ldv, nrots);

	clock_gettime(CLOCK_REALTIME, &finish_time);
	printf("total time: %lld\n", (long long) finish_time.tv_sec - (long long) start_time.tv_sec);


	for (int j = 0; j < n; ++j)
		if (fwrite(A + j * lda, sizeof(double), m, fG) != m)
			error("%s: can't write data to file %s\n", argv[0], argv[4]);

	fclose(fG);
	free(A);

	for (int j = 0; j < n; ++j)
		if (fwrite(V + j * ldv, sizeof(double), n, fV) != n)
			error("%s: can't write data to file %s\n", argv[0], argv[5]);

	fclose(fV);
	free(V);

	free(nrots);

	return 0;
}
