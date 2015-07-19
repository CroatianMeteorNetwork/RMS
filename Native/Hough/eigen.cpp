/*
 * Copyright (C) 2008 Leandro A. F. Fernandes and Manuel M. Oliveira
 *
 * author   : Fernandes, Leandro A. F.
 * e-mail   : laffernandes@gmail.com
 * home page: http://www.inf.ufrgs.br/~laffernandes
 *
 *
 * The complete description of the implemented techinique can be found at
 *
 *      Leandro A. F. Fernandes, Manuel M. Oliveira
 *      Real-time line detection through an improved Hough transform voting scheme
 *      Pattern Recognition (PR), Elsevier, 41:1, 2008, 299-314.
 *      DOI: http://dx.doi.org/10.1016/j.patcog.2007.04.003
 *      Project Page: http://www.inf.ufrgs.br/~laffernandes/kht.html
 *
 * If you use this implementation, please reference the above paper.
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#include "eigen.h"

/* Eigenvectors and Eigenvalues
 * Code from "The Better C Eigenvector Source Code Page"
 * http://www.nauticom.net/www/jdtaft/CEigenBetter.htm
 */
inline
void
tri_diagonalize(const double *Cxd, double *d, double *e, double *A, int L, double tol)
{
	int i, j, k, l;
	double f, g, h, hh;
	for (i = 0; i < L; i++) {
		for (j = 0; j <= i; j++) {
			A[i*L + j] = Cxd[i*L + j];
		}
	}
	for (i = L - 1; i > 0; i--) {
		l = i - 2;
		f = A[i*L + i - 1];
		g = 0.0;
		for (k = 0; k <= l; k++) {
			g += A[i*L + k]*A[i*L + k];
		}
		h = g + f*f;
		if (g <= tol) {
			e[i] = f;
			h = 0.0;
			d[i] = h;
			continue;
		}
		l++;
		g = sqrt(h);
		if (f >= 0.0) g = -g;
		e[i] = g;
		h = h - f*g;
		A[i*L + i - 1] = f - g;
		f = 0.0;
		for (j = 0; j <= l; j++) {
			A[j*L + i] = A[i*L + j]/h;
			g = 0.0;
			for (k = 0; k <= j; k++) {
				g += A[j*L + k]*A[i*L + k];
			}
			for (k = j + 1; k <= l; k++) {
				g += A[k*L + j]*A[i*L + k];				
			}
			e[j] = g/h;
			f += g*A[j*L + i];
		}
		hh = f/(h + h);
		for (j = 0; j <= l; j++) {
			f = A[i*L + j];
			g = e[j] - hh*f;
			e[j] = g;
			for (k = 0; k <= j; k++) {
				A[j*L + k] = A[j*L + k] - f*e[k] - g*A[i*L + k];
			}
		}
		d[i] = h;
	}
	d[0] = e[0] = 0.0;
	for (i = 0; i < L; i++) {
		l = i - 1;
		if (d[i] != 0.0) {
			for (j = 0; j <= l; j++) {
				g = 0.0;
				for (k = 0; k <= l; k++) {
					g += A[i*L + k]*A[k*L + j];
				}
				for (k = 0; k <= l; k++) {
					A[k*L + j] = A[k*L + j] - g*A[k*L + i];
				}
			}
		}
		d[i] = A[i*L + i];
		A[i*L + i] = 1.0;
		for (j = 0; j <= l; j++) {
			A[i*L + j] = A[j*L + i] = 0.0;
		}
	}
}

/* Eigenvectors and Eigenvalues
 * Code from "The Better C Eigenvector Source Code Page"
 * http://www.nauticom.net/www/jdtaft/CEigenBetter.htm
 */
inline
static int
calc_eigenstructure(double *d, double *e, double *A, int L, double macheps)
{
	int i, j, k, l, m;
	double b, c, f, g, h, p, r, s;

	for (i = 1; i < L; i++) e[i - 1] = e[i];
	e[L - 1] = b = f = 0.0;
	for (l = 0; l < L; l++) {
		h = macheps*(fabs(d[l]) + fabs(e[l]));
		if (b < h) b = h;
		for (m = l; m < L; m++) {
			if (fabs(e[m]) <= b) break;
		}
		j = 0;
		if (m != l) do {
			if (j++ == 30) return -1;
			p = (d[l + 1] - d[l])/(2.0*e[l]);
			r = sqrt(p*p + 1);
			h = d[l] - e[l]/(p + (p < 0.0 ? -r : r));
			for (i = l; i < L; i++) d[i] = d[i] - h;
			f += h;
			p = d[m];
			c = 1.0;
			s = 0.0;
			for (i = m - 1; i >= l; i--) {
				g = c*e[i];
				h = c*p;
				if (fabs(p) >= fabs(e[i])) {
					c = e[i]/p;
					r = sqrt(c*c + 1);
					e[i + 1] = s*p*r;
					s = c/r;
					c = 1.0/r;
				} else {
					c = p/e[i];
					r = sqrt(c*c + 1);
					e[i + 1] = s*e[i]*r;
					s = 1.0/r;
					c = c/r;
				}
				p = c*d[i] - s*g;
				d[i + 1] = h + s*(c*g + s*d[i]);
				for (k = 0; k < L; k++) {
					h = A[k*L + i + 1];
					A[k*L + i + 1] = s*A[k*L + i] + c*h;
					A[k*L + i] = c*A[k*L + i] - s*h;
				}
			}
			e[l] = s*p;
			d[l] = c*p;
		} while (fabs(e[l]) > b);
		d[l] = d[l] + f;
	}

	/* order the eigenvectors  */
	for (i = 0; i < L; i++) {
		k = i;
		p = d[i];
		for (j = i + 1; j < L; j++) {
			if (d[j] > p) {
				k = j;
				p = d[j];
			}
		}
		if (k != i) {
			d[k] = d[i];
			d[i] = p;
			for (j = 0; j < L; j++) {
				p = A[j*L + i];
				A[j*L + i] = A[j*L + k];
				A[j*L + k] = p;
			}
		}
	}
	return 0;
}

// Computes the decomposition of a matrix into matrices composed of its eigenvectors and eigenvalues.
void
eigen(matrix_t &vectors, matrix_t &values, const matrix_t &matrix)
{
	double temp[2];

	tri_diagonalize( matrix, values, temp, vectors, 2, 1.0e-6 );
	calc_eigenstructure( values, temp, vectors, 2, 1.0e-16 );

	values[3] = values[1];
	values[1] = values[2] = 0.0;
}
