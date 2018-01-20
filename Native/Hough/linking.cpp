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

#include "linking.h"

// This function complements the linking procedure.
inline
bool
next(int &x_seed, int &y_seed, const unsigned char *binary_image, const size_t image_width, const size_t image_height)
{
	/* Leandro A. F. Fernandes, Manuel M. Oliveira
	 * Real-time line detection through an improved Hough transform voting scheme
	 * Pattern Recognition (PR), Elsevier, 41:1, 2008, 299-314.
	 *
	 * Algorithm 6
	 */

	int x, y;
	static const int X_OFFSET[8] = {  0,  1,  0, -1,  1, -1, -1,  1 };
	static const int Y_OFFSET[8] = {  1,  0, -1,  0,  1,  1, -1, -1 };

    for (size_t i=0; i!=8; ++i)
	{
		x = x_seed + X_OFFSET[i];
		if ((0 <= x) && (x < image_width))
		{
			y = y_seed + Y_OFFSET[i];
			if ((0 <= y) && (y < image_height))
			{
				if (binary_image[y*image_width+x])
				{
					x_seed = x;
					y_seed = y;
					return true;
				}
			}
		}
	}
	return false;
}

// Creates a string of neighboring edge pixels.
inline
void
linking_procedure(string_t &string, unsigned char *binary_image, const size_t image_width, const size_t image_height, const int x_ref, const int y_ref, const double half_width, const double half_height)
{
	/* Leandro A. F. Fernandes, Manuel M. Oliveira
	 * Real-time line detection through an improved Hough transform voting scheme
	 * Pattern Recognition (PR), Elsevier, 41:1, 2008, 299-314.
	 *
	 * Algorithm 5
	 */

	int x, y;

	string.clear();
	
	// Find and add feature pixels to the end of the string.
	x = x_ref;
	y = y_ref;
	do
	{
		pixel_t &p = string.push_back();
		
		p.x_index = x;
		p.y_index = y;

		p.x = x - half_width;
		p.y = y - half_height;

		binary_image[y*image_width+x] = 0;
	}
	while (next( x, y, binary_image, image_width, image_height ));

	pixel_t temp;
	for (size_t i=0, j=string.size()-1; i<j; ++i, --j)
	{
		temp = string[i];
		string[i] = string[j];
		string[j] = temp;
	}

	// Find and add feature pixels to the begin of the string.
	x = x_ref;
	y = y_ref;
	if (next( x, y, binary_image, image_width, image_height ))
	{
		do
		{
			pixel_t &p = string.push_back();

			p.x_index = x;
			p.y_index = y;

			p.x = x - half_width;
			p.y = y - half_height;

			binary_image[y*image_width+x] = 0;
		}
		while (next( x, y, binary_image, image_width, image_height ));
	}
}

// Creates a list of strings of neighboring edge pixels.
void
find_strings(strings_list_t &strings, unsigned char *binary_image, const size_t image_width, const size_t image_height, const size_t min_size)
{
	const double half_width = 0.5 * image_width;
	const double half_height = 0.5 * image_height;

	strings.clear();

	for (int y=1, y_end=image_height-1; y!=y_end; ++y)
	{
		for (int x=1, x_end=image_width-1; x!=x_end; ++x)
		{
			if (binary_image[y*image_width+x])
			{
				string_t &string = strings.push_back();

				linking_procedure( string, binary_image, image_width, image_height, x, y, half_width, half_height );

				if (string.size() < min_size)
				{
					strings.pop_back();
				}
			}
		}
	}
}
