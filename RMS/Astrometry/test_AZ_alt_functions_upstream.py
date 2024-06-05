import sys
sys.path.append('/Users/lucbusquin/Projects/RMS')

import RMS.Astrometry.ApplyAstrometry as aa
import RMS.Formats.Platepar as pp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


platepar = pp.Platepar()
platepar.read("/Users/lucbusquin/Projects/RMS_data/ArchivedFiles/US9999_20240208_013640_169265_detected/platepar_cmn2010.cal")

from RMS.Formats.FFfile import getMiddleTimeFF
from RMS.Astrometry.Conversions import date2JD, jd2Date
time = getMiddleTimeFF('FF_HR000A_20240208_015724_739_0802560.fits', 25)
year, month, day, hour, minute, second, millisecond = time
print(time[0])
jd = date2JD(year, month, day, hour, minute, second, millisecond)

margin = 10
x_grid, y_grid = np.linspace(margin, platepar.X_res-margin, 100), np.linspace(margin, platepar.Y_res-margin, 50)
xx, yy = np.meshgrid(x_grid, y_grid)
xx, yy = xx.flatten(), yy.flatten()

xs = [960, 1800]
ys = [540, 900] 

# Compute celestial coordinates for each grid point
# _, ra_arr, dec_arr, _ = aa.xyToRaDecPP(len(xx)*[platepar.JD], xx, yy, len(xx)*[1], platepar, extinction_correction=False, jd_time=True)
_, ra_arr, dec_arr, _ = aa.xyToRaDecPP(len(xx)*[jd], xs, ys, len(xs)*[1], platepar, extinction_correction=False, jd_time=True)
print(f"center: {ra_arr[0]}, {dec_arr[0]}")
print(f"lower right: {ra_arr[1]}, {dec_arr[1]}")

# # Convert back to pixel coordinates
# x_star, y_star = aa.raDecToXYPP(ra_arr, dec_arr, platepar.JD, platepar)

# # Calculate error for each point and populate the errors matrix
# errors = np.sqrt((x_star - xx)**2 + (y_star - yy)**2)

# errors_reshaped = errors.reshape(len(y_grid), len(x_grid))
# xx_reshaped = xx.reshape(len(y_grid), len(x_grid))
# yy_reshaped = yy.reshape(len(y_grid), len(x_grid))

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Create a surface plot
# surf = ax.plot_surface(xx_reshaped, yy_reshaped, errors_reshaped, cmap='coolwarm',  vmin=0, vmax=.25)


# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.set_zlim(0, .25)
# ax.set_xlim(0, platepar.X_res)
# ax.set_ylim(0, platepar.Y_res)
# ax.set_title('US9999 - XY to RaDEC Roundtrip Error\nUsing Different Parameters in Each Directions')
# ax.set_xlabel('X Coordinate (px)')
# ax.set_ylabel('Y Coordinate(px)')
# ax.set_zlabel('Error(px)')
# plt.gca().invert_yaxis()

# plt.show()

# plt.figure()
# contour = plt.contourf(xx_reshaped, yy_reshaped, errors_reshaped, cmap='coolwarm', levels=100, vmin=0, vmax=0.25)
# plt.colorbar(contour)
# plt.title('US9999 - Error Contour Plot')
# plt.xlabel('X-axis Label')
# plt.ylabel('Y-axis Label')
# plt.gca().invert_yaxis()
# plt.show()


# # Assuming xx, yy, x_star, y_star are all numpy arrays of the same length
# plt.figure(figsize=(10, 6))

# for i in range(len(xx)):
#     dx = x_star[i] - xx[i]
#     dy = y_star[i] - yy[i]
    
#     # Extend the segment by a factor of 10
#     start_x = xx[i] - 9*dx
#     end_x = xx[i] + 9*dx
#     start_y = yy[i] - 9*dy
#     end_y = yy[i] + 9*dy
    
#     plt.plot([start_x, end_x], [start_y, end_y], 'r-')  # 'r-' for red line

# plt.xlabel('X Pixel')
# plt.ylabel('Y Pixel')
# plt.title('Extended Segment Connecting Original and Transformed Coordinates')
# plt.gca().invert_yaxis()
# plt.show()
