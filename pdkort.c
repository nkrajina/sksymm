#include <immintrin.h>
#include <unistd.h>
#include <stdio.h>
#include <mathimf.h>
#include <mkl.h>
#include "pdkort.h"
#include "defs.h"

void pdkgram4(int *p, int *q, int *m, double *A, int *lda, double *a12, double *a13, double *a14, double *a32, double *a42, double *a34)
{
	__assume_aligned(A, 2 * VECLEN * sizeof(double));
	__assume(*lda % (2 * VECLEN) == 0);

	double s12[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double s13[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double s14[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double s32[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double s42[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double s34[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));

	#pragma omp simd simdlen(VECLEN)
	for (int i = 0; i < VECLEN; ++i) {
		s12[i] = 0.0;
		s13[i] = 0.0;
		s14[i] = 0.0;
		s32[i] = 0.0;
		s42[i] = 0.0;
		s34[i] = 0.0;
	}

	double *c1 = A + (2 * *p + 0) * *lda;
	double *c2 = A + (2 * *p + 1) * *lda;
	double *c3 = A + (2 * *q + 0) * *lda;
	double *c4 = A + (2 * *q + 1) * *lda;

	for (int offset = 0; offset < *m; offset += 2 * VECLEN) {
		#pragma omp simd aligned(c1,c2,c3,c4: VECLEN * sizeof(double)) simdlen(VECLEN)
		for (int j = 0; j < VECLEN; ++j) {
			s12[j] += c1[offset + 2 * j + 1] * c2[offset + 2 * j + 0] - c1[offset + 2 * j + 0] * c2[offset + 2 * j + 1];
			s13[j] += c1[offset + 2 * j + 1] * c3[offset + 2 * j + 0] - c1[offset + 2 * j + 0] * c3[offset + 2 * j + 1];
			s14[j] += c1[offset + 2 * j + 1] * c4[offset + 2 * j + 0] - c1[offset + 2 * j + 0] * c4[offset + 2 * j + 1];
			s32[j] += c3[offset + 2 * j + 1] * c2[offset + 2 * j + 0] - c3[offset + 2 * j + 0] * c2[offset + 2 * j + 1];
			s42[j] += c4[offset + 2 * j + 1] * c2[offset + 2 * j + 0] - c4[offset + 2 * j + 0] * c2[offset + 2 * j + 1];
			s34[j] += c3[offset + 2 * j + 1] * c4[offset + 2 * j + 0] - c3[offset + 2 * j + 0] * c4[offset + 2 * j + 1];
		 }
	}

	*a12 = 0.0;
	*a13 = 0.0;
	*a14 = 0.0;
	*a32 = 0.0;
	*a42 = 0.0;
	*a34 = 0.0;

	#pragma omp simd simdlen(VECLEN)
	for (int j = 0; j < VECLEN; ++j) {
		*a12 += s12[j];
		*a13 += s13[j];
		*a14 += s14[j];
		*a32 += s32[j];
		*a42 += s42[j];
		*a34 += s34[j];
	}
}

void dgesv2(double *a11, double *a12, double *a21, double *a22, double *snr, double *csr, double *snl, double *csl)
{
	double f, g, h;
	double z;
	double csg, sng;

	int swap;

	double l, t, m, s, r, a;
	double tv, sv, cv, su, cu;

	// QR factorization of A
	swap = (fabs(*a11) < fabs(*a21));

	if (!swap) {
		f = *a11;
		g = *a21;
	} else {
		f = *a21;
		g = *a11;
	}

	z = (f != 0.0 ? g / f : 0.0);

	if (!swap) {
		csg = 1.0 / sqrt(1.0 + z * z);
		sng = z * csg;
	} else {
		sng = 1.0 / sqrt(1.0 + z * z);
		csg = z * sng;
	}

	*a11 = f * sqrt(1.0 + z * z);

	f = *a12;
	g = *a22;

	*a12 =  csg * f + sng * g;
	*a22 = -sng * f + csg * g;

	// SVD of G' * A
	swap = (fabs(*a11) < fabs(*a22));

	if (!swap) {
		f = *a11;
		g = *a12;
		h = *a22;
	} else {
		f =  *a22;
		g = -*a12;
		h =  *a11;
	}

	if (g == 0.0) {
		cv = 1.0;
		sv = 0.0;
		cu = 1.0;
		su = 0.0;
	} else if (f == 0.0) {
		// in this case h == 0, so swap was not performed
		cv = 1.0;
		sv = 0.0;
		cu = 0.0;
		su = 1.0;

		*a11 = *a12;
		*a22 = 0.0;
	} else {
		l = 1.0 - fabs(h) / fabs(f);
		t = 2.0 - l;
		m = g / f;
		s = sqrt(t * t + m * m);
		r = sqrt(l * l + m * m);
		a = 0.5 * (r + s);

		*a11 = fabs(f) * a;
		*a22 = (f >= 0 ? h / a : -h / a);

		tv = 0.5 * (m / (s + t) + m / (r + l)) * (1 + a);
		cv = 1.0 / sqrt(1.0 + tv * tv);
		sv = tv * cv;
		cu = *a11 / f * cv;
		su = *a22 / f * sv;
	}

	if (!swap) {
		*csr = cv;
		*snr = sv;
		*csl = csg * cu - sng * su;
		*snl = sng * cu + csg * su;
	} else {
		*csr = -su;
		*snr =  cu;
		*csl = -csg * sv - sng * cv;
		*snl = -sng * sv + csg * cv;
	}
}

void pdkupd4(int *p, int *q, int *m, double *A, int *lda, double *snr1, double *csr1, double *snl1, double *csl1, double *snr2, double *csr2, double *snl2, double *csl2)
{
	__assume_aligned(A, 2 * VECLEN * sizeof(double));
	__assume(*lda % (2 * VECLEN) == 0);

	double *c1 = A + (2 * *p + 0) * *lda;
	double *c2 = A + (2 * *p + 1) * *lda;
	double *c3 = A + (2 * *q + 0) * *lda;
	double *c4 = A + (2 * *q + 1) * *lda;

	for (int offset = 0; offset < *m; offset += VECLEN) {
		double T[VECLEN * 4] __attribute__((aligned(VECLEN * sizeof(double))));

		#pragma omp simd aligned(c1, c2, c3, c4, T: VECLEN * sizeof(double)) simdlen(VECLEN)
		for (int j = 0; j < VECLEN; ++j) {
			T[0 * VECLEN + j] = c1[offset + j];
			T[1 * VECLEN + j] = c2[offset + j];
			T[2 * VECLEN + j] = c3[offset + j];
			T[3 * VECLEN + j] = c4[offset + j];
		}

		double l[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
		double r[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));

		//	drot(m, c1, &one, c3, &one, &cs1, &sn1);
		#pragma omp simd aligned(T: VECLEN * sizeof(double)) simdlen(VECLEN)
		for (int j = 0; j < VECLEN; ++j) {
			l[j] =  *csl1 * T[0 * VECLEN + j] + *snl1 * T[2 * VECLEN + j];
			r[j] = -*snl1 * T[0 * VECLEN + j] + *csl1 * T[2 * VECLEN + j];

			T[0 * VECLEN + j] = l[j];
			T[2 * VECLEN + j] = r[j];
		}

		//	drot(m, c2, &one, c4, &one, &cs2, &sn2);
		#pragma omp simd aligned(T: VECLEN * sizeof(double)) simdlen(VECLEN)
		for (int j = 0; j < VECLEN; ++j) {
			l[j] =  *csr1 * T[1 * VECLEN + j] + *snr1 * T[3 * VECLEN + j];
			r[j] = -*snr1 * T[1 * VECLEN + j] + *csr1 * T[3 * VECLEN + j];

			T[1 * VECLEN + j] = l[j];
			T[3 * VECLEN + j] = r[j];
		}

		//	drot(m, c1, &one, c4, &one, &cs3, &sn3);
		#pragma omp simd aligned(T: VECLEN * sizeof(double)) simdlen(VECLEN)
		for (int j = 0; j < VECLEN; ++j) {
			l[j] =  *csl2 * T[0 * VECLEN + j] + *snl2 * T[3 * VECLEN + j];
			r[j] = -*snl2 * T[0 * VECLEN + j] + *csl2 * T[3 * VECLEN + j];

			T[0 * VECLEN + j] = l[j];
			T[3 * VECLEN + j] = r[j];
		}

		//	drot(m, c2, &one, c3, &one, &cs4, &sn4);
		#pragma omp simd aligned(T: VECLEN * sizeof(double)) simdlen(VECLEN)
		for (int j = 0; j < VECLEN; ++j) {
			l[j] =  *csr2 * T[1 * VECLEN + j] + *snr2 * T[2 * VECLEN + j];
			r[j] = -*snr2 * T[1 * VECLEN + j] + *csr2 * T[2 * VECLEN + j];

			T[1 * VECLEN + j] = l[j];
			T[2 * VECLEN + j] = r[j];
		}

		#pragma omp simd aligned(c1, c2, c3, c4, A: VECLEN * sizeof(double)) simdlen(VECLEN)
		for (int j = 0; j < VECLEN; ++j) {
			c1[offset + j] = T[0 * VECLEN + j];
			c2[offset + j] = T[1 * VECLEN + j];
			c3[offset + j] = T[2 * VECLEN + j];
			c4[offset + j] = T[3 * VECLEN + j];
		}
	}
}

int pdkort4(int *p, int *q, int *m, int *n, double *A, int *lda, double *V, int *ldv)
{
	__assume_aligned(A, 2 * VECLEN * sizeof(double));
	__assume_aligned(V, 2 * VECLEN * sizeof(double));

	__assume(*lda % (2 * VECLEN) == 0);
	__assume(*ldv % (2 * VECLEN) == 0);

	double a12, a13, a14, a32, a42, a34;
	double snr1, csr1, snl1, csl1, snr2, csr2, snl2, csl2;

	pdkgram4(p, q, m, A, lda, &a12, &a13, &a14, &a32, &a42, &a34);

	dgesv2(&a12, &a14, &a32, &a34, &snr1, &csr1, &snl1, &csl1);
	a34 = -a34;
	dgesv2(&a12, &a13, &a42, &a34, &snr2, &csr2, &snl2, &csl2);

	pdkupd4(p, q, m, A, lda, &snr1, &csr1, &snl1, &csl1, &snr2, &csr2, &snl2, &csl2);
	pdkupd4(p, q, n, V, ldv, &snr1, &csr1, &snl1, &csl1, &snr2, &csr2, &snl2, &csl2);

	if (fabs(csr1) == 1.0 && fabs(csl1) == 1.0 && fabs(csr2) == 1.0 && fabs(csr2) == 1.0)
		return 0;

	return 1;
}

void dgesv2_vec(double *a11, double *a12, double *a21, double *a22, double *snr, double *csr, double *snl, double *csl)
{
	__assume_aligned(a11, VECLEN * sizeof(double));
	__assume_aligned(a12, VECLEN * sizeof(double));
	__assume_aligned(a21, VECLEN * sizeof(double));
	__assume_aligned(a22, VECLEN * sizeof(double));
	__assume_aligned(snr, VECLEN * sizeof(double));
	__assume_aligned(csr, VECLEN * sizeof(double));
	__assume_aligned(snl, VECLEN * sizeof(double));
	__assume_aligned(csl, VECLEN * sizeof(double));

	double f[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double g[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double h[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double z[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double csg[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double sng[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));

	int swap[VECLEN] __attribute__((aligned(VECLEN * sizeof(int))));

	double l[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double t[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double m[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double s[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double r[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double a[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double tv[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double sv[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double cv[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double su[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double cu[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));


	#pragma omp simd simdlen(VECLEN)
	for (int i = 0; i < VECLEN; ++i) {
		// QR factorization of A
		swap[i] = (fabs(a11[i]) < fabs(a21[i]));

		if (!swap[i]) {
			f[i] = a11[i];
			g[i] = a21[i];
		} else {
			f[i] = a21[i];
			g[i] = a11[i];
		}

		z[i] = (f[i] != 0.0 ? g[i] / f[i] : 0.0);

		if (!swap[i]) {
			csg[i] = 1.0 / sqrt(1.0 + z[i] * z[i]);
			sng[i] = z[i] * csg[i];
		} else {
			sng[i] = 1.0 / sqrt(1.0 + z[i] * z[i]);
			csg[i] = z[i] * sng[i];
		}

		a11[i] = f[i] * sqrt(1.0 + z[i] * z[i]);

		f[i] = a12[i];
		g[i] = a22[i];

		a12[i] =  csg[i] * f[i] + sng[i] * g[i];
		a22[i] = -sng[i] * f[i] + csg[i] * g[i];

		// SVD of G' * A
		swap[i] = (fabs(a11[i]) < fabs(a22[i]));

		if (!swap[i]) {
			f[i] = a11[i];
			g[i] = a12[i];
			h[i] = a22[i];
		} else {
			f[i] =  a22[i];
			g[i] = -a12[i];
			h[i] =  a11[i];
		}

		if (g[i] == 0.0) {
			cv[i] = 1.0;
			sv[i] = 0.0;
			cu[i] = 1.0;
			su[i] = 0.0;
		} else if (f[i] == 0.0) {
			// in this case h == 0, so swap was not performed
			cv[i] = 1.0;
			sv[i] = 0.0;
			cu[i] = 0.0;
			su[i] = 1.0;

			a11[i] = a12[i];
			a22[i] = 0.0;
		} else {
			l[i] = 1.0 - fabs(h[i]) / fabs(f[i]);
			t[i] = 2.0 - l[i];
			m[i] = g[i] / f[i];
			s[i] = sqrt(t[i] * t[i] + m[i] * m[i]);
			r[i] = sqrt(l[i] * l[i] + m[i] * m[i]);
			a[i] = 0.5 * (r[i] + s[i]);

			a11[i] = fabs(f[i]) * a[i];
			a22[i] = (f[i] >= 0 ? h[i] / a[i] : -h[i] / a[i]);

			tv[i] = 0.5 * (m[i] / (s[i] + t[i]) + m[i] / (r[i] + l[i])) * (1.0 + a[i]);
			cv[i] = 1.0 / sqrt(1.0 + tv[i] * tv[i]);
			sv[i] = tv[i] * cv[i];
			cu[i] = a11[i] / f[i] * cv[i];
			su[i] = a22[i] / f[i] * sv[i];
		}

		if (!swap[i]) {
			csr[i] = cv[i];
			snr[i] = sv[i];
			csl[i] = csg[i] * cu[i] - sng[i] * su[i];
			snl[i] = sng[i] * cu[i] + csg[i] * su[i];
		} else {
			csr[i] = -su[i];
			snr[i] =  cu[i];
			csl[i] = -csg[i] * sv[i] - sng[i] * cv[i];
			snl[i] = -sng[i] * sv[i] + csg[i] * cv[i];
		}
	}
}

int pdkort4_vec(int *p, int *q, int *m, int *n, double *A, int *lda, double *V, int *ldv)
{
	__assume_aligned(A, 2 * VECLEN * sizeof(double));
	__assume_aligned(V, 2 * VECLEN * sizeof(double));
	__assume_aligned(p, VECLEN * sizeof(int));
	__assume_aligned(q, VECLEN * sizeof(int));
	__assume(*lda % (2 * VECLEN) == 0);
	__assume(*ldv % (2 * VECLEN) == 0);

	double a12[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double a13[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double a14[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double a32[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double a42[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double a34[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));

	double snl1[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double csl1[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double snr1[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double csr1[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double snl2[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double csl2[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double snr2[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));
	double csr2[VECLEN] __attribute__((aligned(VECLEN * sizeof(double))));

	for (int i = 0; i < VECLEN; ++i)
		pdkgram4(&p[i], &q[i], m, A, lda, &a12[i], &a13[i], &a14[i], &a32[i], &a42[i], &a34[i]);

	#pragma forceinline
	dgesv2_vec(a12, a14, a32, a34, snr1, csr1, snl1, csl1);

	#pragma omp simd simdlen(VECLEN)
	for (int i = 0; i < VECLEN; ++i)
		a34[i] = -a34[i];

	#pragma forceinline
	dgesv2_vec(a12, a13, a42, a34, snr2, csr2, snl2, csl2);

	for (int i = 0; i < VECLEN; ++i) {
		pdkupd4(&p[i], &q[i], m, A, lda, &snr1[i], &csr1[i], &snl1[i], &csl1[i], &snr2[i], &csr2[i], &snl2[i], &csl2[i]);
		pdkupd4(&p[i], &q[i], n, V, ldv, &snr1[i], &csr1[i], &snl1[i], &csl1[i], &snr2[i], &csr2[i], &snl2[i], &csl2[i]);
	}

	int nrots = 0;

	for (int i = 0; i < VECLEN; ++i)
		nrots += (fabs(csr1[i]) != 1.0 || fabs(csr2[i]) != 1.0 || fabs(csl1[i]) != 1.0 && fabs(csl2[i]) != 1.0 ? 1 : 0);

	return nrots;
}

