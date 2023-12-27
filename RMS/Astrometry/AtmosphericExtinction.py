""" Function for computing atmospheric extinction. """

import numpy as np

def atmosphericExtinctionCorrection(elev, ht):
	""" Compute the magnitude loss due to atmospheric extinction. Extinction was assumed for the human eye,
		as described in Green (1992).

	Argument:
		elev: [float] Elevation of the object above horizon (deg).
		ht: [float] Height above sea level (m).

	Return:
		[float] Corrected magnitude.

	"""

	# Compute the zenith angle
	z = np.radians(90.0 - elev)


	# Convert height to km
	h = ht/1000

	# Compute the air mass
	x = (np.cos(z) + 0.025*np.exp(-11*np.cos(z)))**-1

	# Rayleigh scattering magnitude loss
	a_ray = 0.1451*np.exp(-h/7.996)

	# Aerosol scattering
	a_aer = 0.120*np.exp(-h/1.5)

	# Ozone scattering
	a_oz = 0.016


	# Compute total magnitude loss
	a_tot = a_ray + a_aer + a_oz

	# Return the total magnitude loss
	return a_tot*x



if __name__ == "__main__":

	import matplotlib.pyplot as plt

	# Plot absolute extinctions for difference elevations and heights above sea level
	ht_list = [0, 0.5, 1, 2, 3]

	for ht in ht_list:

		elev_list = np.linspace(0, 90, 1000)

		plt.plot(elev_list, atmosphericExtinctionCorrection(elev_list, 1000*ht), label='Ht = {:.1f} km'.format(ht))


	plt.legend()

	plt.xlabel('Elevation (deg)')
	plt.ylabel('Absolute extinction (mag)')

	plt.show()



	# Plot relative extinctions for difference elevations and heights above sea level
	ht_list = [0, 0.5, 1, 2, 3]

	for ht in ht_list:

		elev_list = np.linspace(0, 90, 1000)

		# Compute relative extinction
		rel_exticntion = atmosphericExtinctionCorrection(elev_list, 1000*ht) - atmosphericExtinctionCorrection(90, 1000*ht)

		plt.plot(elev_list, rel_exticntion, label='Ht = {:.1f} km'.format(ht))


	plt.legend()

	plt.xlabel('Elevation (deg)')
	plt.ylabel('Relative extinction (mag)')

	plt.show()

