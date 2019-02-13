""" Functions for reading/writing UFOOrbit files. """

import os



def writeUFOOrbit(dir_path, file_name, data):
	""" Write the UFOObit input CSV file. 

	Arguments:
		dir_path: [str] Directory where the file will be written to.
		file_name: [str] Name of the UFOOrbit CSV file.
		data: [list] A list of meteor entries.

	"""


	with open(os.path.join(dir_path, file_name), 'w') as f:

		# Write the header
		f.write("Ver,Y,M,D,h,m,s,Mag,Dur,Az1,Alt1,Az2,Alt2, Ra1, Dec1, Ra2, Dec2,ID,Long,Lat,Alt,Tz\n")

		# Write meteor data to file
		for line in data:

			dt, peak_mag, duration, azim1, alt1, azim2, alt2, ra1, dec1, ra2, dec2, cam_code, lon, lat, \
				elev, UT_corr = line

			# Convert azimuths to the astronomical system (+W of due S)
			azim1 = (azim1 - 180)%360
			azim2 = (azim2 - 180)%360


			# Extract date and time from the datetime object
			dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000


			f.write('{:s},{:4d},{:2d},{:2d},{:2d},{:2d},{:4.2f},{:.2f},{:.3f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:.7f},{:s},{:.6f},{:.6f},{:.1f},{:.1f},\n'.format(\
				'R91', dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond/1000000, \
				peak_mag, duration, azim1, alt1, azim2, alt2, ra1, dec1, ra2, dec2, cam_code, lon, lat, \
				elev, UT_corr))
