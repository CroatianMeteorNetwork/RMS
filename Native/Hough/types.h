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

#ifndef _TYPES_
#define _TYPES_

#include <cmath>
#include <cstdlib>
#include <memory.h>
#include "buffer_2d.h"

// A simple accumulator class implementation.
class accumulator_t
{
public:

	// Lower and upper bounds definition.
	struct bounds_t
	{
		double lower;
		double upper;

		bounds_t(double _lower, double _upper)
		{
			lower = _lower;
			upper = _upper;
		}
	};

private:

	// Accumulator bins ([1,height][1,width] range).
	int **m_bins;

	// Discretization step.
	double m_delta;
	
	// Accumulator height (theta dimention).
	size_t m_height;

	// Expected image height.
	size_t m_image_height;
	
	// Expected image width.
	size_t m_image_width;

	// Specifies the discretization of the rho dimention ([1,width] range).
	double *m_rho;

	// Parameters space limits (rho dimention, in pixels).
	bounds_t m_rho_bounds;

	// Specifies the size of allocated storage for the accumulator (rho dimention).
	size_t m_rho_capacity;

	// Specifies the discretization of the theta dimention ([1,height] range).
	double *m_theta;
	
	// Parameters space limits (theta dimentions, in degrees).
	bounds_t m_theta_bounds;

	// Specifies the size of allocated storage for the accumulator (theta dimention).
	size_t m_theta_capacity;

	// Accumulator width (rho dimention).
	size_t m_width;

public:

	// Class constructor.
	accumulator_t() :
		m_bins(0),
		m_delta(0),
		m_height(0),
		m_image_height(0),
		m_image_width(0),
		m_rho(0),
		m_rho_bounds(0,0),
		m_rho_capacity(0),
		m_theta(0),
		m_theta_bounds(0,0),
		m_theta_capacity(0),
		m_width(0)
	{
	}
	
	// Class constructor.
	accumulator_t(const size_t image_width, const size_t image_height, const double delta) :
		m_bins(0),
		m_delta(0),
		m_height(0),
		m_image_height(0),
		m_image_width(0),
		m_rho(0),
		m_rho_bounds(0,0),
		m_rho_capacity(0),
		m_theta(0),
		m_theta_bounds(0,0),
		m_theta_capacity(0),
		m_width(0)
	{
		init( image_width, image_height, delta );
	}

	// Class destructor.
	~accumulator_t()
	{
		free( m_bins );
		free( m_rho );
		free( m_theta );
	}

	// Returns the accumulator bins ([1,height][1,width] range).
	inline
	int** bins()
	{
		return m_bins;
	}

	// Returns the accumulator bins ([1,height][1,width] range).
	inline
	const int** bins() const
	{
		return const_cast<const int**>( m_bins );
	}

	// Set zeros to the accumulator bins.
	inline
	void clear()
	{
		memset_2d( m_bins, 0, m_height + 2, m_width + 2, sizeof( int ) );
	}

	// Returns the discretization step.
	inline
	double delta() const
	{
		return m_delta;
	}

	// Returns the accumulator height (theta dimention).
	inline
	size_t height() const
	{
		return m_height;
	}

	// Returns the expected image height.
	inline
	size_t image_height() const
	{
		return m_image_height;
	}

	// Returns the expected image width.
	inline
	size_t image_width() const
	{
		return m_image_width;
	}

	// Initializes the accumulator.
	inline
	void init(const size_t image_width, const size_t image_height, const double delta)
	{
		if ((m_delta != delta) || (m_image_width != image_width) || (m_image_height != image_height))
		{
			m_delta = delta;
			m_image_width = image_width;
			m_image_height = image_height;

			// Rho.
			double r = sqrt( static_cast<double>( (image_width * image_width) + (image_height * image_height) ) );

			m_width = static_cast<size_t>( (r + 1.0) / delta );

			if (m_rho_capacity < (m_width + 2))
			{
				m_rho = static_cast<double*>( realloc( m_rho, (m_rho_capacity = (m_width + 2)) * sizeof( double ) ) );
			}

			m_rho[1] = -0.5 * r;
			for (size_t i=2; i<=m_width; ++i)
			{
				m_rho[i] = m_rho[i-1] + delta;
			}
			
			m_rho_bounds.lower = -0.5 * r;
			m_rho_bounds.upper =  0.5 * r;

			// Theta.
			m_height = static_cast<size_t>( 180.0 / delta );

			if (m_theta_capacity < (m_height + 2))
			{
				m_theta = static_cast<double*>( realloc( m_theta, (m_theta_capacity = (m_height + 2)) * sizeof( double ) ) );
			}

			m_theta[1] = 0.0;
			for (size_t i=2; i<=m_height; i++)
			{
				m_theta[i] = m_theta[i-1] + delta;
			}

			m_theta_bounds.lower = 0.0;
			m_theta_bounds.upper = 180.0 - delta;

			// Accumulator bins.
			m_bins = static_cast<int**>( realloc_2d( m_bins, m_height + 2, m_width + 2, sizeof( int ) ) );
		}

		clear();
	}
	
	// Returns the discretization of the rho dimention (in pixels, [1,width] range).
	inline
	const double* rho() const
	{
		return m_rho;
	}

	// Returns the parameters space limits (rho dimention, in pixels).
	inline
	const bounds_t& rho_bounds() const
	{
		return m_rho_bounds;
	}

	// Returns the discretization of the theta dimention (in degrees, [1,height] range).
	inline
	const double* theta() const
	{
		return m_theta;
	}

	// Returns the parameters space limits (theta dimentions, in degrees).
	inline
	const bounds_t& theta_bounds() const
	{
		return m_theta_bounds;
	}

	// Returns the accumulator width (rho dimention).
	inline
	size_t width() const
	{
		return m_width;
	}
};

// A simple list implementation (use it only with aggregate types).
template<typename item_type, size_t capacity_inc>
class list
{
private:

	// Specifies the size of allocated storage for the container.
	size_t m_capacity;

	// Specifies the list of items.
	item_type *m_items;

	// Counts the number of elements.
	size_t m_size;

public:

	// Erases the elements of the list.
	inline
	void clear()
	{
		m_size = 0;
	}

	// Tests if the list is empty.
	inline
	bool empty() const
	{
		return (m_size == 0);
	}

	// Returns a pointer to the list of items.
	inline
	item_type* items()
	{
		return m_items;
	}
	
	// Returns a pointer to the list of items.
	inline
	const item_type* items() const
	{
		return m_items;
	}
	
	// Class constructor.
	list() :
		m_capacity(0),
		m_items(0),
		m_size(0)
	{
	}

	// Class destructor.
	~list()
	{
		free( m_items );
	};

	// Deletes the element at the end of the list.
	inline
	void pop_back()
	{
		m_size--;
	}
	
	// Adds a new last element and returns a reference to it.
	inline
	item_type& push_back()
	{
		if (m_capacity == m_size)
		{
			m_items = (item_type*)realloc( m_items, (m_capacity += capacity_inc) * sizeof( item_type ) );
			memset( &m_items[m_size], 0, capacity_inc * sizeof( item_type ) );
		}
		return m_items[m_size++];
	}

	// Specifies a new capacity for a list.
	inline
	void reserve(const size_t capacity)
	{
		if (m_capacity < capacity)
		{
			size_t first = m_capacity;
			m_items = (item_type*)realloc( m_items, (m_capacity = capacity) * sizeof( item_type ) );
			memset( &m_items[first], 0, (capacity - first) * sizeof( item_type ) );
		}

		if (m_size > capacity)
		{
			m_size = capacity;
		}
	}

	// Specifies a new size for a list.
	inline
	void resize(const size_t size)
	{
		if (m_capacity < size)
		{
			size_t first = m_capacity;
			m_items = (item_type*)realloc( m_items, (m_capacity = size) * sizeof( item_type ) );
			memset( &m_items[first], 0, (size - first) * sizeof( item_type ) );
		}
		m_size = size;
	}

	// Returns the number of elements.
	inline
	size_t size() const
	{
		return m_size;
	}

	// Returns a reference to the list element at a specified position.
	inline
	item_type& operator [] (const size_t index)
	{
		return m_items[index];
	}

	// Returns a reference to the list element at a specified position.
	inline
	const item_type& operator [] (const size_t index) const
	{
		return m_items[index];
	}
};

// A feature pixel.
struct pixel_t
{
	int x_index;
	int y_index;

	double x;
	double y;
};

// A cluster of approximately collinear feature pixels.
struct cluster_t
{
	const pixel_t *pixels;
	size_t size;
};

// A 2D line (normal equation parameters).
struct line_t
{
	double rho;
	double theta;
};

// A 2D point.
struct point_t
{
	double x;
	double y;
};

// Specifies a list of approximately collinear feature pixels.
typedef list<cluster_t,1000> clusters_list_t;

// Specifies a list of lines.
typedef list<line_t,1000> lines_list_t;

// A 2x2 matrix.
typedef double matrix_t[4];

// Specifies a string of adjacent feature pixels.
typedef list<pixel_t,500> string_t;

// Specifies a list of string of feature pixels.
typedef list<string_t,1000> strings_list_t;

#endif // !_TYPES_
