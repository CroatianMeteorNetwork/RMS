""" Plots points on an all-sky sinusoidal projection plot. """


import matplotlib.pyplot as plt


import numpy as np


class AllSkyPlot(object):
	def __init__(self, ra0=180.0):

		self.ra0 = ra0

		self.fig = plt.figure()

		# Set background color
		self.fig.patch.set_facecolor('black')
		self.ax = self.fig.add_subplot(1, 1, 1, facecolor='black')

		# Set equal aspect ratio
		self.ax.set_aspect('equal')

		# # Set tick color
		# self.ax.tick_params(axis='x', colors='0.5')
		# self.ax.tick_params(axis='y', colors='0.5')

		# Turn off ticks
		self.ax.tick_params(labeltop=False, labelright=False, labelbottom=False, labelleft=False)

		self.plotGrid()


	def raDec2XY(self, ra, dec):

		# Normalize
		x = (ra - self.ra0)*np.cos(np.radians(dec))
		y = dec

		return x, y


	def plotGrid(self, step=15):


		# Plot a meridian and parallel grid
		ra_grid = np.arange(0, 360 + step, step)
		dec_grid = np.arange(-90, 90 + step, step)


		# Plot meridians
		for ra in ra_grid:

			# Increase number of points for meridian plot so they are smoother
			step_finer = step/5
			dec_arr = np.arange(-90, 90 + step_finer, step_finer)

			ra_temp = np.zeros_like(dec_arr) + ra

			x_grid, y_grid = self.raDec2XY(ra_temp, dec_arr)

			self.ax.plot(x_grid, y_grid, linestyle='dotted', alpha=0.5, color='gray')


		# Plot parallels
		for dec in dec_grid:

			dec_temp = np.zeros_like(ra_grid) + dec

			x_grid, y_grid = self.raDec2XY(ra_grid, dec_temp)

			self.ax.plot(x_grid, y_grid, linestyle='dotted', alpha=0.5, color='gray')


		# Plot ticks
		for dec in dec_grid:

			x, y = self.raDec2XY(self.ra0 - 180, dec)

			if dec > 0:
				va = 'bottom'

			else:
				va = 'top'

			self.ax.text(x, y, "{:+d}$^\circ$".format(dec), color='0.5', ha='right', va=va, size=7)

		# Plot every other RA tick and skip 0 and 360
		for ra in ra_grid[1:-1:2]:

			x, y = self.raDec2XY(ra, 0)
			self.ax.text(x, y, "{:+d}$^\circ$".format(ra), color='0.5', ha='center', va='top', size=7)




	
	def beautify(self):

		self.ax.set_xlim([-180, 180])
		self.ax.set_ylim([-90, 90])

		self.fig.tight_layout()


	def show(self):

		self.beautify()
		plt.show()






if __name__ == "__main__":

	allsky_plot = AllSkyPlot()

	allsky_plot.show()