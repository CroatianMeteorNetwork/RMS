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

#ifndef _KHT_
#define _KHT_

#include "types.h"

/* Kernel-based Hough transform (KHT) for detecting straight lines in images.
 *
 * This function performs the KHT procedure over a given binary image and returns a
 * list with the [rho theta] parameters (rho in pixels and theta in degrees) of the
 * detected lines. This implementation assumes that the binary image was obtained
 * using, for instance, a Canny edge detector plus thresholding and thinning.
 *
 * The resulting lines are in the form:
 *
 *     rho = x * cos(theta) + y * sin(theta)
 *
 * and we assume that the origin of the image coordinate system is at the center of
 * the image, with the x-axis growing to the right and the y-axis growing down.
 *
 * The function parameters are:
 *
 *                 'lines' : This list will be populated with detected lines, sorted
 *                           in descending order of relevance.
 *
 *          'binary_image' : Input binary image buffer (single channel), where 0
 *                           denotes black and 1 to 255 denotes feature pixels.
 *
 *           'image_width' : Image width.
 *
 *          'image_height' : Image height.
 *
 *      'cluster_min_size' : Minimum number of pixels in the clusters of approximately
 *                           collinear feature pixels. The default value is 10.
 *
 * 'cluster_min_deviation' : Minimum accepted distance between a feature pixel and
 *                           the line segment defined by the end points of its cluster.
 *                           The default value is 2.
 *
 *                 'delta' : Discretization step for the parameter space. The default
 *                           value is 0.5.
 *
 *     'kernel_min_height' : Minimum height for a kernel pass the culling operation.
 *                           This property is restricted to the [0,1] range. The
 *                           default value is 0.002.
 *
 *              'n_sigmas' : Number of standard deviations used by the Gaussian kernel
 *                           The default value is 2.
 *
 * It is important to notice that the linking procedure implemented by the kht()
 * function destroys the original image.
 */
void kht(lines_list_t &lines, unsigned char *binary_image, const size_t image_width, const size_t image_height, const size_t cluster_min_size = 10, const double cluster_min_deviation = 2.0, const double delta = 0.5, const double kernel_min_height = 0.002, const double n_sigmas = 2.0);

#endif // !_KHT_
