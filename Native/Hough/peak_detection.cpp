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

#include "buffer_2d.h"
#include "peak_detection.h"

// The coordinates of a bin of the accumulator.
struct bin_t
{
	size_t rho_index;   // [1,rho_size] range.
	size_t theta_index; // [1,theta_size] range.

	int votes;
};

// Specifies a list of accumulator bins.
typedef list<bin_t,1000> bins_list_t;

// An auxiliar data structure that identifies which accumulator bin was visited by the peak detection procedure.
class visited_map_t
{
private:

	// The map of flags ([1,theta_size][1,rho_size] range).
	bool **m_map;

	// Specifies the size of allocated storage for the map (rho dimention).
	size_t m_rho_capacity;

	// Specifies the size of allocated storage for the map (theta dimention).
	size_t m_theta_capacity;

public:

	// Initializes the map.
	inline
	void init(const size_t accumulator_width, const size_t accumulator_height)
	{
		if ((m_rho_capacity < (accumulator_width + 2)) || (m_theta_capacity < (accumulator_height + 2)))
		{
			m_rho_capacity = accumulator_width + 2;
			m_theta_capacity = accumulator_height + 2;

			m_map = static_cast<bool**>( realloc_2d( m_map, m_theta_capacity, m_rho_capacity, sizeof( bool ) ) );
		}

		memset_2d( m_map, 0, m_theta_capacity, m_rho_capacity, sizeof( bool ) );
	}
	
	// Sets a given accumulator bin as visited.
	inline
	void set_visited(const size_t rho_index, size_t theta_index)
	{
		m_map[theta_index][rho_index] = true;
	}

	// Class constructor.
	visited_map_t() :
		m_map(0),
		m_rho_capacity(0),
		m_theta_capacity(0)
	{
	}

	// Class destructor()
	~visited_map_t()
	{
		free( m_map );
	}

	// Returns whether a neighbour bin was visited already.
	inline
	bool visited_neighbour(const size_t rho_index, const size_t theta_index) const
	{
		return m_map[theta_index-1][rho_index-1] || m_map[theta_index-1][rho_index  ] || m_map[theta_index-1][rho_index+1] ||
			   m_map[theta_index  ][rho_index-1] ||                                      m_map[theta_index  ][rho_index+1] ||
			   m_map[theta_index+1][rho_index-1] || m_map[theta_index+1][rho_index  ] || m_map[theta_index+1][rho_index+1];
	}
};

inline
int
compare_bins(const bin_t *bin1, const bin_t *bin2)
{
	return (bin1->votes < bin2->votes) ? 1 : ((bin1->votes > bin2->votes) ? -1 : 0);
}

// Computes the convolution of the given cell with a (discrete) 3x3 Gaussian kernel.
inline
int
convolution(const int **bins, const int rho_index, const int theta_index)
{
	return bins[theta_index-1][rho_index-1] + bins[theta_index-1][rho_index+1] + bins[theta_index+1][rho_index-1] + bins[theta_index+1][rho_index+1] +
	       bins[theta_index-1][rho_index  ] + bins[theta_index-1][rho_index  ] + bins[theta_index  ][rho_index-1] + bins[theta_index  ][rho_index-1] + bins[theta_index  ][rho_index+1] + bins[theta_index  ][rho_index+1] + bins[theta_index+1][rho_index  ] + bins[theta_index+1][rho_index  ] +
	       bins[theta_index  ][rho_index  ] + bins[theta_index  ][rho_index  ] + bins[theta_index  ][rho_index  ] + bins[theta_index  ][rho_index  ];
}

// Identify the peaks of votes (most significant straight lines) in the accmulator.
void
peak_detection(lines_list_t &lines, const accumulator_t &accumulator)
{
	/* Leandro A. F. Fernandes, Manuel M. Oliveira
	 * Real-time line detection through an improved Hough transform voting scheme
	 * Pattern Recognition (PR), Elsevier, 41:1, 2008, 299-314.
	 *
	 * Section 3.4
	 */

	const int **bins = accumulator.bins();
	const double *rho = accumulator.rho();
	const double *theta = accumulator.theta();

	// Create a list with all cells that receive at least one vote.
	static bins_list_t used_bins;
	
	size_t used_bins_count = 0;
	for (size_t theta_index=1, theta_end=accumulator.height()+1; theta_index!=theta_end; ++theta_index)
	{
		for (size_t rho_index=1, rho_end=accumulator.width()+1; rho_index!=rho_end; ++rho_index)
		{
			if (bins[theta_index][rho_index])
			{
				used_bins_count++;
			}
		}
	}
	used_bins.resize( used_bins_count );
		
	for (size_t theta_index=1, i=0, theta_end=accumulator.height()+1; theta_index!=theta_end; ++theta_index)
	{
		for (size_t rho_index=1, rho_end=accumulator.width()+1; rho_index!=rho_end; ++rho_index)
		{
			if (bins[theta_index][rho_index])
			{
				bin_t &bin = used_bins[i];

				bin.rho_index = rho_index;
				bin.theta_index = theta_index;
				bin.votes = convolution( bins, rho_index, theta_index ); // Convolution of the cells with a 3x3 Gaussian kernel

				i++;
			}
		}
	}
	
	// Sort the list in descending order according to the result of the convolution.
	std::qsort( used_bins.items(), used_bins_count, sizeof( bin_t ), (int(*)(const void*, const void*))compare_bins );
	
	// Use a sweep plane that visits each cell of the list.
	static visited_map_t visited;
	visited.init( accumulator.width(), accumulator.height() );

	lines.clear();
	lines.reserve( used_bins_count );

	for (size_t i=0; i!=used_bins_count; ++i)
	{
		bin_t &bin = used_bins[i];

		if (!visited.visited_neighbour( bin.rho_index, bin.theta_index ))
		{
			line_t &line = lines.push_back();
			
			line.rho = rho[bin.rho_index];
			line.theta = theta[bin.theta_index];
		}
		visited.set_visited( bin.rho_index, bin.theta_index );
	}
}
