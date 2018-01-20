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

#include <limits>
#include "voting.h"
#include "eigen.h"

// pi value.
static const double pi = 3.14159265358979323846;

// An elliptical-Gaussian kernel.
struct kernel_t
{
	const cluster_t *pcluster;

	double rho;
	double theta;

	matrix_t lambda;    // [sigma^2_rho sigma_rhotheta; sigma_rhotheta sigma^2_theta]

	size_t rho_index;	// [1,rho_size] range
	size_t theta_index; // [1,theta_size] range

	double height;
};

// Specifies a list of Gaussian kernels.
typedef list<kernel_t,1000> kernels_list_t;

// Specifies a list of pointers to Gaussian kernels.
typedef list<kernel_t*,1000> pkernels_list_t;

// Bi-variated Gaussian distribution.
inline
double
gauss(const double rho, const double theta, const double sigma2_rho, const double sigma2_theta, const double sigma_rho_sigma_theta, const double two_r, const double a, const double b)
{
	double z = ((rho * rho) / sigma2_rho) - ((two_r * rho * theta) / sigma_rho_sigma_theta) + ((theta * theta) / sigma2_theta);
	return a * exp( -z * b );
}

// Bi-variated Gaussian distribution.
inline
double
gauss(const double rho, const double theta, const double sigma2_rho, const double sigma2_theta, const double sigma_rho_theta)
{
	/* Leandro A. F. Fernandes, Manuel M. Oliveira
	 * Real-time line detection through an improved Hough transform voting scheme
	 * Pattern Recognition (PR), Elsevier, 41:1, 2008, 299-314.
	 *
	 * Equation 15
	 */

	const double sigma_rho_sigma_theta = sqrt( sigma2_rho ) * sqrt( sigma2_theta );
	const double r = (sigma_rho_theta / sigma_rho_sigma_theta), two_r = 2.0 * r;
	const double a = 1.0 / (2.0 * pi * sigma_rho_sigma_theta * sqrt( 1.0 - (r * r) ));
	const double b = 1.0 / (2.0 * (1.0 - (r * r)));
	return gauss( rho, theta, sigma2_rho, sigma2_theta, sigma_rho_sigma_theta, two_r, a, b );
}

// Solves the uncertainty propagation.
inline
void
solve(matrix_t &result, const matrix_t &nabla, const matrix_t &lambda)
{
	matrix_t result1 = {}, result2 = {};

	for (size_t i=0, i_line=0; i<2; ++i, i_line+=2)
	{
		for (size_t j=0; j<2; ++j)
		{
			for (size_t k=0; k<2; ++k)
			{
				result1[i_line+j] += nabla[i_line+k] * lambda[(k*2)+j];
			}
		}
	}

	for (size_t i=0, i_line=0; i<2; ++i, i_line+=2)
	{
		for (size_t j=0; j<2; ++j)
		{
			for (size_t k=0; k<2; ++k)
			{
				result2[i_line+j] += result1[i_line+k] * nabla[(j*2)+k];
			}
		}
	}

	memcpy( result, result2, 4 * sizeof( double ) );
}

// This function complements the proposed voting process.
inline
void
vote(accumulator_t &accumulator, size_t rho_start_index, const size_t theta_start_index, const double rho_start, const double theta_start, int inc_rho_index, const int inc_theta_index, const double sigma2_rho, const double sigma2_theta, const double sigma_rho_theta, const double scale)
{
	/* Leandro A. F. Fernandes, Manuel M. Oliveira
	 * Real-time line detection through an improved Hough transform voting scheme
	 * Pattern Recognition (PR), Elsevier, 41:1, 2008, 299-314.
	 *
	 * Algorithm 4
	 */

	int **bins = accumulator.bins();
	
	const size_t rho_size = accumulator.width(), theta_size = accumulator.height();
	const double delta = accumulator.delta();
	const double inc_rho = delta * inc_rho_index, inc_theta = delta * inc_theta_index;
		
	const double sigma_rho_sigma_theta = sqrt( sigma2_rho ) * sqrt( sigma2_theta );
	const double r = (sigma_rho_theta / sigma_rho_sigma_theta), two_r = 2.0 * r;
	const double a = 1.0 / (2.0 * pi * sigma_rho_sigma_theta * sqrt( 1.0 - (r * r) ));
	const double b = 1.0 / (2.0 * (1.0 - (r * r)));

	bool theta_voted;
	double rho, theta;
	int votes, theta_not_voted = 0;
	size_t rho_index, theta_index, theta_count = 0;
	
	// Loop for the theta coordinates of the parameter space.
	theta_index = theta_start_index;
	theta = theta_start;
	do
	{
		// Test if the kernel exceeds the parameter space limits.
		if ((theta_index == 0) || (theta_index == (theta_size + 1)))
		{
			rho_start_index = rho_size - rho_start_index + 1;
			theta_index = (theta_index == 0) ? theta_size : 1;
			inc_rho_index = -inc_rho_index;
		}

		// Loop for the rho coordinates of the parameter space.
		theta_voted = false;

		rho_index = rho_start_index;
		rho = rho_start;
		while (((votes = static_cast<int>( (gauss( rho, theta, sigma2_rho, sigma2_theta, sigma_rho_sigma_theta, two_r, a, b ) * scale) + 0.5 )) > 0) && (rho_index >= 1) && (rho_index <= rho_size))
		{
			bins[theta_index][rho_index] += votes;
			theta_voted = true;

			rho_index += inc_rho_index;
			rho += inc_rho;
		}

		if (!theta_voted)
		{
			theta_not_voted++;
		}

		theta_index += inc_theta_index;
		theta += inc_theta;
		theta_count++;
	}
	while ((theta_not_voted != 2) && (theta_count < theta_size));
}

// Performs the proposed Hough transform voting scheme.
void
voting(accumulator_t &accumulator, const clusters_list_t &clusters, const double kernel_min_height, const double n_sigmas)
{
	/* Leandro A. F. Fernandes, Manuel M. Oliveira
	 * Real-time line detection through an improved Hough transform voting scheme
	 * Pattern Recognition (PR), Elsevier, 41:1, 2008, 299-314.
	 *
	 * Algorithm 2
	 */
	static kernels_list_t kernels;
	static pkernels_list_t used_kernels;

	kernels.resize( clusters.size() );
	used_kernels.resize( clusters.size() );

	matrix_t M, V, S;
	point_t mean, u, v;
	double x, y, Sxx, Syy, Sxy, aux;

	static const double rad_to_deg = 180.0 / pi;
	const double delta = accumulator.delta();
	const double one_div_delta = 1.0 / delta;
	const double n_sigmas2 = n_sigmas * n_sigmas;
	const double rho_max = accumulator.rho_bounds().upper;

	for (size_t k=0, end=clusters.size(); k!=end; ++k)
	{
		const cluster_t &cluster = clusters[k];
		kernel_t &kernel = kernels[k];

		kernel.pcluster = &cluster;
		
		// Alternative reference system definition.
		mean.x = mean.y = 0.0;
		for (size_t i=0; i!=cluster.size; ++i)
		{
			mean.x += cluster.pixels[i].x;
			mean.y += cluster.pixels[i].y;
		}
		mean.x /= cluster.size;
		mean.y /= cluster.size;
		
		Sxx = Syy = Sxy = 0.0;
		for (size_t i=0; i!=cluster.size; ++i)
		{
			x = cluster.pixels[i].x - mean.x;
			y = cluster.pixels[i].y - mean.y;
		
			Sxx += (x * x);
			Syy += (y * y);
			Sxy += (x * y);
		}
		
		M[0] = Sxx;
		M[3] = Syy;
		M[1] = M[2] = Sxy;
		eigen( V, S, M );

		u.x = V[0];
		u.y = V[2];
		
		v.x = V[1];
		v.y = V[3];
		
		// y_v >= 0 condition verification.
		if (v.y < 0.0)
		{
			v.x *= -1.0;
			v.y *= -1.0;
		}

		// Normal equation parameters computation (Eq. 3).
		kernel.rho = (v.x * mean.x) + (v.y * mean.y);
		kernel.theta = acos( v.x ) * rad_to_deg;

		kernel.rho_index = static_cast<size_t>( std::abs( (kernel.rho + rho_max) * one_div_delta ) ) + 1;
		kernel.theta_index = static_cast<size_t>( std::abs( kernel.theta * one_div_delta ) ) + 1;

		// sigma^2_m' and sigma^2_b' computation, substituting Eq. 5 in Eq. 10.
		aux = sqrt( 1.0 - (v.x * v.x) );
		matrix_t nabla = {
				             -((u.x * mean.x) + (u.y * mean.y)), 1.0,
				(aux != 0.0) ? ((u.x / aux) * rad_to_deg) : 0.0, 0.0
			};

		aux = 0.0;
		for (size_t i=0; i!=cluster.size; ++i)
		{
			x = (u.x * (cluster.pixels[i].x - mean.x)) + (u.y * (cluster.pixels[i].y - mean.y));
			aux += (x * x);
		}

		matrix_t lambda = {
				1.0 / aux,                0.0,
					  0.0, 1.0 / cluster.size
			};

		// Uncertainty from sigma^2_m' and sigma^2_b' to sigma^2_rho,  sigma^2_theta and sigma_rho_theta.
		solve( kernel.lambda, nabla, lambda );

		if (kernel.lambda[3] == 0.0)
		{
			kernel.lambda[3] = 0.1;
		}

		kernel.lambda[0] *= n_sigmas2;
		kernel.lambda[3] *= n_sigmas2;

		// Compute the height of the kernel.
		kernel.height = gauss( 0.0, 0.0, kernel.lambda[0], kernel.lambda[3], kernel.lambda[1] );
	}

	/* Leandro A. F. Fernandes, Manuel M. Oliveira
	 * Real-time line detection through an improved Hough transform voting scheme
	 * Pattern Recognition (PR), Elsevier, 41:1, 2008, 299-314.
	 *
	 * Algorithm 3
	 */

	// Discard groups with very short kernels.
	double norm = std::numeric_limits<double>::min();

	for (size_t k=0, end=kernels.size(); k!=end; ++k)
	{
		kernel_t &kernel = kernels[k];

		if (norm < kernel.height)
		{
			norm = kernel.height;
		}
		used_kernels[k] = &kernel;
	}
	norm = 1.0 / norm;

	size_t i = 0;
	for (size_t k=0, end=used_kernels.size(); k!=end; ++k)
	{
		if ((used_kernels[k]->height * norm) >= kernel_min_height)
		{
			if (i != k)
			{
				kernel_t *temp = used_kernels[i];
				used_kernels[i] = used_kernels[k];
				used_kernels[k] = temp;
			}
			i++;
		}
	}
	used_kernels.resize( i );

	// Find the g_min threshold and compute the scale factor for integer votes.
	double radius, scale;
	double kernels_scale = std::numeric_limits<double>::min();

	for (size_t k=0, end=used_kernels.size(); k!=end; ++k)
	{
		kernel_t &kernel = *used_kernels[k];

		eigen( V, S, kernel.lambda );
		radius = sqrt( S[3] );

		scale = gauss( V[1] * radius, V[3] * radius, kernel.lambda[0], kernel.lambda[3], kernel.lambda[1] );
		scale = (scale < 1.0) ? (1.0 / scale) : 1.0;

		if (kernels_scale < scale)
		{
			kernels_scale = scale;
		}
	}

	// Vote for each selected kernel.
	for (size_t k=0, end=used_kernels.size(); k!=end; ++k)
	{
		kernel_t &kernel = *used_kernels[k];

		vote( accumulator, kernel.rho_index,     kernel.theta_index,        0.0,    0.0,  1,  1, kernel.lambda[0], kernel.lambda[3], kernel.lambda[1], kernels_scale );
		vote( accumulator, kernel.rho_index,     kernel.theta_index - 1,    0.0, -delta,  1, -1, kernel.lambda[0], kernel.lambda[3], kernel.lambda[1], kernels_scale );
		vote( accumulator, kernel.rho_index - 1, kernel.theta_index,     -delta,    0.0, -1,  1, kernel.lambda[0], kernel.lambda[3], kernel.lambda[1], kernels_scale );
		vote( accumulator, kernel.rho_index - 1, kernel.theta_index - 1, -delta, -delta, -1, -1, kernel.lambda[0], kernel.lambda[3], kernel.lambda[1], kernels_scale );
	}
}
