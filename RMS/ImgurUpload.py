""" Upload an image to imgur and get a link to the image. """

from __future__ import print_function, division, absolute_import

# Imgur RMS client ID
CLIENT_ID = "ca85d1ed0b3fa85"


import base64
import json

try:
    import urllib.request as urllib2
    import urllib.parse as urllib
except ImportError:
    import urllib2
    import urllib



def imgurUpload(file_path, image_data=None):
	""" Upload the given image to Imgur. 
	
	Arguments:
		file_path: [str] Path to the image file.

	Keyword arguments:
		image_data: [bytes] Read in image in JPG, PNG, etc. format.

	Return:
		img_url: [str] URL to the uploaded image.
	"""

	# Read the image if image data was not given
	if image_data is None:
		
		# Open the image in binary mode
		f = open(file_path, "rb")
		image_data = f.read()


	# Encode the image
	b64_image = base64.standard_b64encode(image_data)


	# Upload the image
	headers = {'Authorization': 'Client-ID ' + CLIENT_ID}
	data = {'image': b64_image, 'title': 'test'} # create a dictionary.

	request = urllib2.Request(url="https://api.imgur.com/3/upload.json", 
		data=urllib.urlencode(data).encode("utf-8"), headers=headers)
	response = urllib2.urlopen(request).read()


	# Get URL to image
	parse = json.loads(response)
	img_url = parse['data']['link']
	

	return img_url




if __name__ == "__main__":

	print(imgurUpload("/home/dvida/Desktop/FF_BR0001_20190204_234235_646_0000256.fits_maxpixel.jpg"))