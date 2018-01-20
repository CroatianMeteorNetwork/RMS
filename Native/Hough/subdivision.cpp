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

#include <algorithm>
#include "subdivision.h"

// Subdivides the string of feature pixels into sets of most perceptually significant straight line segments.
inline
double
subdivision_procedure(clusters_list_t &clusters, const string_t &string, const size_t first_index, const size_t last_index, const double min_deviation, const size_t min_size)
{
	/* D. G. Lowe
	 * Three-dimensional object recognition from single two-dimensional images
	 * Artificial Intelligence, Elsevier, 31, 1987, 355–395.
	 *
	 * Section 4.6
	 */

	size_t clusters_count = clusters.size();

	const pixel_t &first = string[first_index];
	const pixel_t &last = string[last_index];
	
	// Compute the length of the straight line segment defined by the endpoints of the cluster.
	int x = first.x_index - last.x_index;
	int y = first.y_index - last.y_index;
	double length = sqrt( static_cast<double>( (x * x) + (y * y) ) );
	
	// Find the pixels with maximum deviation from the line segment in order to subdivide the cluster.
	size_t max_pixel_index = 0;
	double deviation, max_deviation = -1.0;

	for (size_t i=first_index, count=string.size(); i!=last_index; i=(i+1)%count)
	{
		const pixel_t &current = string[i];
		
		deviation = static_cast<double>( abs( ((current.x_index - first.x_index) * (first.y_index - last.y_index)) + ((current.y_index - first.y_index) * (last.x_index - first.x_index)) ) );

		if (deviation > max_deviation)
		{
			max_pixel_index = i;
			max_deviation = deviation;
		}
	}
	max_deviation /= length;

	// Compute the ratio between the length of the segment and the maximum deviation.
	double ratio = length / std::max( max_deviation, min_deviation );

	// Test the number of pixels of the sub-clusters.
	if (((max_pixel_index - first_index + 1) >= min_size) && ((last_index - max_pixel_index + 1) >= min_size))
	{
		double ratio1 = subdivision_procedure( clusters, string, first_index, max_pixel_index, min_deviation, min_size );
		double ratio2 = subdivision_procedure( clusters, string, max_pixel_index, last_index, min_deviation, min_size );

		// Test the quality of the sub-clusters against the quality of the current cluster.
		if ((ratio1 > ratio) || (ratio2 > ratio))
		{
			return std::max( ratio1, ratio2 );
		}
	}

	// Remove the sub-clusters from the list of clusters.
	clusters.resize( clusters_count );

	// Keep current cluster
	cluster_t &cluster = clusters.push_back();
	
	cluster.pixels = &first;
	cluster.size = (last_index - first_index) + 1;

	return ratio;
}

// Creates a list of clusters of approximately collinear feature pixels.
void
find_clusters(clusters_list_t &clusters, const strings_list_t &strings, const double min_deviation, const size_t min_size)
{
	clusters.clear();

	for (size_t i=0, end=strings.size(); i!=end; ++i)
	{
		const string_t &string = strings[i];
		subdivision_procedure( clusters, string, 0, string.size() - 1, min_deviation, min_size );
	}
}
