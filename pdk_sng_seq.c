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

	while (1) {
		sweep_rots = 0;

		// each of nblocks steps consists of floor(nblocks / 2) orthogonalizations performed VECLEN at the time
		for (int s = 0; s < nblocks; ++s)
			for (int j = 0; j < nblocks / 2; ++j) {
				int p, q;

				modulus(j, s, nblocks, &p, &q);
				sweep_rots += pdkort4(&p, &q, m, n, A, lda, V, ldv);
			}

		total_rots += sweep_rots;
		++nsweeps;

		printf("Sweep %d: %d\n", nsweeps, sweep_rots);

		if (sweep_rots == 0)
			break;
	}

	return total_rots;
}
