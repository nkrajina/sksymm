#include <mkl.h>
#include <stdio.h>
#include <omp.h>
#include "pdk.h"
#include "pdkort.h"

void modulus_even(int k, int s, int n, int *i, int *j)
{
	if (0 <= k && k <= s / 2 - 1) {
		*i = k;
		*j = s - 1 - *i;
	} else if ((s + 1) / 2 <= k && k <= n / 2 - 1) {
		*j = n / 2 + k;
		*i = n + s - 1 - *j;
	} else {
		*i = s / 2;
		*j = n / 2 + s / 2;
	}
}

void modulus_odd(int k, int s, int n, int *i, int *j)
{
	if (0 <= k && k <= s / 2 - 1) {
		*i = k;
		*j = s - 1 - *i;
	} else {
		*j = (n + 1) / 2 + k;
		*i = n + s - 1 - *j;
	}
}

int pdk(int *m, int *n, double *A, int *lda, double *V, int *ldv, int *nrots)
{
	__assume_aligned(nrots, VECLEN * sizeof(int));

	int nblocks = *n / 2;
	void (*modulus)(int, int, int, int *, int *) = (nblocks % 2 ? modulus_odd : modulus_even);

	int sweep_rots;
	int total_rots = 0;
	int nsweeps = 0;

	if (nblocks == 1)
		return 0;
	else if (nblocks == 2) {
		int p = 0;
		int q = 1;

		pdkort4(&p, &q, m, n, A, lda, V, ldv);
		return 1;
	}

	#pragma omp parallel shared(sweep_rots,nrots)
	{
		int tix = omp_get_thread_num();
		int nthreads = omp_get_num_threads();

		while (1) {
			#pragma omp master
			sweep_rots = 0;

			nrots[tix] = 0;
			for (int s = 0; s < nblocks; ++s)
				#pragma omp for
				for (int offset = 0; offset < nblocks / 2; offset += VECLEN)
					if (offset + VECLEN < nblocks / 2) {
						int p[VECLEN] __attribute__((aligned(VECLEN * sizeof(int))));
						int q[VECLEN] __attribute__((aligned(VECLEN * sizeof(int))));

						#pragma omp simd simdlen(VECLEN)
						for (int j = 0; j < VECLEN; ++j)
							modulus(offset + j, s, nblocks, &p[j], &q[j]);

						nrots[tix] += pdkort4_vec(p, q, m, n, A, lda, V, ldv);
					} else {
						for (int j = offset; j < nblocks / 2; ++j) {
							int p, q;

							modulus(j, s, nblocks, &p, &q);
							nrots[tix] += pdkort4(&p, &q, m, n, A, lda, V, ldv);
						}
					}

			#pragma omp master
			{
				for (int i = 0; i < nthreads; ++i)
					sweep_rots += nrots[i];

				total_rots += sweep_rots;
				++nsweeps;

				printf("Sweep %d: %d\n", nsweeps, sweep_rots);
			}

			#pragma omp barrier

			if (sweep_rots == 0)
				break;

			#pragma omp barrier
		}
	}

	return total_rots;
}
