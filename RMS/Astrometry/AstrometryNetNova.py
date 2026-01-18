#!/usr/bin/env python

""" 
Astrometry.net client script for communicating with astrometry.net servers.
Modified from: https://github.com/dstndstn/astrometry.net/blob/master/net/client/client.py
"""

from __future__ import print_function

import os
import sys
import copy
import time
import base64
import json
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.application  import MIMEApplication
from email.encoders import encode_noop

import warnings

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.utils.exceptions import AstropyWarning

try:
    # Python 2
    import StringIO
    BytesIO = StringIO.StringIO

except ImportError:
    # Python 3
    import io
    BytesIO = io.BytesIO


try:
    # py3
    from urllib.parse import urlparse, urlencode, quote
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
except ImportError:
    # py2
    from urlparse import urlparse
    from urllib import urlencode, quote
    from urllib2 import urlopen, Request, HTTPError


from PIL import Image


from RMS.Formats.FFfile import read as readFF
from RMS.ImgurUpload import imgurUpload



# Denis' API key for nova.astrometry.net
API_KEY = "sybwjtfjbrpgomep"

# Astrometry.net API URLs
NOVA_API_URL = 'http://nova.astrometry.net/api/'
CONTRAILCAST_API_URL = 'https://astro.contrailcast.com/api/'

# Primary server (contrailcast is faster and more reliable)
PRIMARY_API_URL = CONTRAILCAST_API_URL
FALLBACK_API_URL = NOVA_API_URL

DEBUG = False


def printDebug(*args):
    if DEBUG:
        print(*args)


def json2python(data):
    try:
        return json.loads(data)
    except:
        pass
    return None

python2json = json.dumps

class MalformedResponse(Exception):
    pass
class RequestError(Exception):
    pass

class Client(object):
    default_url = 'http://nova.astrometry.net/api/'

    def __init__(self,
                 apiurl = default_url):
        self.session = None
        self.apiurl = apiurl

    def get_url(self, service):
        return self.apiurl + service

    def send_request(self, service, args={}, file_args=None):
        '''
        service: string
        args: dict
        '''
        if self.session is not None:
            args.update({ 'session' : self.session })
        printDebug('Python:', args)
        json = python2json(args)
        printDebug('Sending json:', json)
        url = self.get_url(service)
        printDebug('Sending to URL:', url)

        # If we're sending a file, format a multipart/form-data
        if file_args is not None:
            import random
            boundary_key = ''.join([random.choice('0123456789') for i in range(19)])
            boundary = '===============%s==' % boundary_key
            headers = {'Content-Type':
                       'multipart/form-data; boundary="%s"' % boundary}
            data_pre = (
                '--' + boundary + '\n' +
                'Content-Type: text/plain\r\n' +
                'MIME-Version: 1.0\r\n' +
                'Content-disposition: form-data; name="request-json"\r\n' +
                '\r\n' +
                json + '\n' +
                '--' + boundary + '\n' +
                'Content-Type: application/octet-stream\r\n' +
                'MIME-Version: 1.0\r\n' +
                'Content-disposition: form-data; name="file"; filename="%s"' % file_args[0] +
                '\r\n' + '\r\n')
            data_post = (
                '\n' + '--' + boundary + '--\n')
            data = data_pre.encode() + file_args[1] + data_post.encode()

        else:
            # Else send x-www-form-encoded
            data = {'request-json': json}
            printDebug('Sending form data:', data)
            data = urlencode(data)
            data = data.encode('utf-8')
            printDebug('Sending data:', data)
            headers = {}

        request = Request(url=url, headers=headers, data=data)

        try:
            f = urlopen(request)
            txt = f.read()
            printDebug('Got json:', txt)
            result = json2python(txt)
            printDebug('Got result:', result)
            stat = result.get('status')
            printDebug('Got status:', stat)
            if stat == 'error':
                errstr = result.get('errormessage', '(none)')
                raise RequestError('server error message: ' + errstr)
            return result
        except HTTPError as e:
            printDebug('HTTPError', e)
            txt = e.read()
            open('err.html', 'wb').write(txt)
            printDebug('Wrote error text to err.html')

    def login(self, apikey):
        args = { 'apikey' : apikey }
        result = self.send_request('login', args)
        if result is not None:
            sess = result.get('session')
            printDebug('Got session:', sess)
            if not sess:
                raise RequestError('no session in result')
            self.session = sess
        else:
            raise RequestError('no session in result')


    def _get_upload_args(self, **kwargs):
        args = {}
        for key,default,typ in [('allow_commercial_use', 'd', str),
                                ('allow_modifications', 'd', str),
                                ('publicly_visible', 'y', str),
                                ('scale_units', None, str),
                                ('scale_type', None, str),
                                ('scale_lower', None, float),
                                ('scale_upper', None, float),
                                ('scale_est', None, float),
                                ('scale_err', None, float),
                                ('center_ra', None, float),
                                ('center_dec', None, float),
                                ('parity',None,int),
                                ('radius', None, float),
                                ('downsample_factor', None, int),
                                ('positional_error', None, float),
                                ('tweak_order', None, int),
                                ('crpix_center', None, bool),
                                ('x', None, list),
                                ('y', None, list),
                                ('image_width', None, int),
                                ('image_height', None, int),
                                ]:
            if key in kwargs:
                val = kwargs.pop(key)
                val = typ(val)
                args.update({key: val})
            elif default is not None:
                args.update({key: default})
        printDebug('Upload args:', args)
        return args

    def url_upload(self, url, **kwargs):
        args = dict(url=url)
        args.update(self._get_upload_args(**kwargs))
        result = self.send_request('url_upload', args)
        return result

    def upload(self, fn=None, img_data=None, **kwargs):
        args = self._get_upload_args(**kwargs)
        file_args = None
        
        if fn is not None:
            try:
                f = open(fn, 'rb')
                file_args = (os.path.basename(fn), f.read())
            except IOError:
                printDebug('File %s does not exist' % fn)
                raise
        
        if img_data is not None:
            file_args = ('image.png', img_data)

        return self.send_request('upload', args, file_args)

    def submission_images(self, subid):
        result = self.send_request('submission_images', {'subid':subid})
        return result.get('image_ids')

    def myjobs(self):
        result = self.send_request('myjobs/')
        return result['jobs']

    def job_status(self, job_id, justdict=False):
        result = self.send_request('jobs/%s' % job_id)
        if justdict:
            return result
        stat = result.get('status')
        if stat == 'success':
            result = self.send_request('jobs/%s/calibration' % job_id)
            printDebug('Calibration:', result)
            result = self.send_request('jobs/%s/tags' % job_id)
            printDebug('Tags:', result)
            result = self.send_request('jobs/%s/machine_tags' % job_id)
            printDebug('Machine Tags:', result)
            result = self.send_request('jobs/%s/objects_in_field' % job_id)
            printDebug('Objects in field:', result)
            result = self.send_request('jobs/%s/annotations' % job_id)
            printDebug('Annotations:', result)
            result = self.send_request('jobs/%s/info' % job_id)
            printDebug('Calibration:', result)

        return stat

    def sub_status(self, sub_id, justdict=False):
        result = self.send_request('submissions/%s' % sub_id)
        if justdict:
            return result
        return result.get('status')

    def jobs_by_tag(self, tag, exact):
        exact_option = 'exact=yes' if exact else ''
        result = self.send_request(
            'jobs_by_tag?query=%s&%s' % (quote(tag.strip()), exact_option),
            {},
        )
        return result



def novaAstrometryNetSolve(ff_file_path=None, img=None, x_data=None, y_data=None, fov_w_range=None,
    api_key=None, x_center=None, y_center=None, api_url=None):
    """ Find an astrometric solution of X, Y image coordinates of stars detected on an image using the
        nova.astrometry.net service or a compatible API.

    Keyword arguments:
        ff_file_path: [str] Path to the FF file to load.
        img: [ndarray] Numpy array containing image data.
        x_data: [list] A list of star x image coordinates.
        y_data: [list] A list of star y image coordinates
        fov_w_range: [2 element tuple] A tuple of scale_lower and scale_upper, i.e. the estimate of the
            width of the FOV in degrees.
        api_key: [str] nova.astrometry.net user API key. None by default, in which case the default API
            key will be used.
        x_center: [float] X coordinate of the image center. If not given, the image center will be used.
        y_center: [float] Y coordinate of the image center. If not given, the image center will be used.
        api_url: [str] Custom API URL. None by default, in which case nova.astrometry.net will be used.
            Can be set to use alternative servers like 'https://astro.contrailcast.com/api/'.

    Return:
        (ra, dec, orientation, scale, fov_w, fov_h, star_data): [tuple of floats] All in degrees,
            scale in px/deg.
    """


    def _printWebLink(stat, first_status=None):

        if first_status is not None:
            if not len(stat.get("user_images", "")):
                stat = first_status

        if len(stat.get("user_images", "")):
            # Use correct server URL for the web link
            is_nova = api_url is None or 'nova.astrometry.net' in api_url
            if is_nova:
                base_url = "http://nova.astrometry.net"
            else:
                # Extract base URL from api_url (remove /api/ suffix)
                base_url = api_url.rstrip('/').replace('/api', '')
            print("Link to web page: {}/user_images/{:d}".format(base_url, stat.get("user_images", "")[0]))


    # Read the FF file, if given
    if ff_file_path is not None:
        
        # Read the FF file
        ff = readFF(*os.path.split(ff_file_path))

        img = ff.avepixel

    else:
        file_handle = None


    tmpimg = None
    img_data = None
    image_url = None

    # Convert an image to a file handle
    if img is not None:

        # Save the avepixel as a memory file
        file_handle = BytesIO()
        pil_img = Image.fromarray(img)

        # Save image to memory as JPG
        pil_img.save(file_handle, format='JPEG')
        img_data = file_handle.getvalue()

    # Create client with custom URL if provided
    if api_url is not None:
        c = Client(apiurl=api_url)
        # For custom servers (like contrailcast), upload directly - don't use imgur
        use_direct_upload = True
    else:
        c = Client()
        # For nova.astrometry.net, use imgur URL upload (their preferred method)
        use_direct_upload = False
        if img_data is not None:
            try:
                image_url = imgurUpload('skyfit_image.jpg', image_data=img_data)
            except Exception as e:
                # Imgur failed, fall back to direct upload
                use_direct_upload = True

    # If direct upload needed, save to temp file
    if use_direct_upload and img_data is not None and image_url is None:
        tmpimg = os.path.join(os.getenv('TMP', default='/tmp'), 'skyfit_image.png')
        pil_img.save(tmpimg)

    # Log in to the astrometry service
    if api_key is None:
        api_key = API_KEY

    c.login(api_key)

    # Add keyword arguments
    kwargs = {}
    kwargs['publicly_visible'] = 'y'
    kwargs['crpix_center'] = True
    kwargs['tweak_order'] = 3

    # Add the scale to keyword arguments, if given
    if fov_w_range is not None:
        scale_lower, scale_upper = fov_w_range
        kwargs['scale_lower'] = scale_lower
        kwargs['scale_upper'] = scale_upper
        kwargs['scale_units'] = 'degwidth'  # FOV range is in degrees


    # Upload image or the list of stars
    if file_handle is not None:
        if image_url is not None:
            # Use URL upload (for nova.astrometry.net via imgur)
            upres = c.url_upload(image_url, **kwargs)
        elif img_data is not None:
            # Direct upload with image data in memory (for contrailcast)
            upres = c.upload(img_data=img_data, **kwargs)
        elif tmpimg is not None:
            # Upload from temp file
            upres = c.upload(fn=tmpimg, **kwargs)
        else:
            upres = None

    elif x_data is not None:
        # For coordinate-only uploads, include image dimensions if available
        if x_center is not None and y_center is not None:
            kwargs['image_width'] = int(x_center * 2)
            kwargs['image_height'] = int(y_center * 2)
        upres = c.upload(x=x_data, y=y_data, **kwargs)

    else:
        upres = None
        print('No input given to the function!')


    if upres is None:
        print('Upload failed!')
        return None

    stat = upres['status']
    if stat != 'success':
        
        print('Upload failed: status', stat)
        print(upres)

        return False

    # Submission ID
    sub_id = upres['subid']

    # Wait until the plate is solved
    solution_tries = 20
    tries = 0
    while True:

        # Limit the number of checking if the field is solved, so the script does not get stuck
        if tries > solution_tries:
            _printWebLink(stat)
            return None
        
        stat = c.sub_status(sub_id, justdict=True)
        print('Got status:', stat)

        if stat is not None:

            jobs = stat.get('jobs', [])
            
            if len(jobs):

                for j in jobs:
                    if j is not None:
                        break

                if j is not None:
                    print('Selecting job id', j)
                    solved_id = j
                    break

        time.sleep(5)

        tries += 1


    first_status = copy.deepcopy(stat)

    # Print the link to the web page
    print("Astrometry.net link:")
    _printWebLink(stat, first_status=first_status)

    # Get results
    get_results_tries = 10
    get_solution_tries = 15  # ~75 sec polling, enough for 1 min server timeout
    results_tries = 0
    solution_tries = 0
    while True:

        # Limit the number of tries of getting the results, so the script does not get stuck
        if results_tries > get_results_tries:
            print('Too many tries in getting the results!')
            _printWebLink(stat, first_status=first_status)
            return None

        if solution_tries > get_solution_tries:
            print('Waiting too long for the solution!')
            _printWebLink(stat, first_status=first_status)
            return None

        # Get the job status
        stat = c.job_status(solved_id, justdict=True)

        # Check if the solution is done
        if stat.get('status','') in ['success']:
            
            # Get the calibration
            result = c.send_request('jobs/%s/calibration' % solved_id)
            print(result)
            break

        elif stat.get('status','') in ['failure']:
            print('Failed to find a solution!')

            _printWebLink(stat, first_status=first_status)

            return None

        # Wait until the job is solved
        elif stat.get('status','') in ['solving']:
            print('Solving... Try {:d}/{:d}'.format(solution_tries, get_solution_tries))
            time.sleep(5)
            solution_tries += 1
            continue

        # Print other error messages
        else:
            time.sleep(5)
            print('Got job status:', stat)
            results_tries += 1


    print()
    # Extract the job ID
    print('Job ID:', solved_id)

    # Download the wcs.fits file
    print("Downloading the WCS file...")
    # Use the correct server URL for WCS download
    # Nova uses numeric job IDs, custom servers use UUIDs
    is_nova = api_url is None or 'nova.astrometry.net' in api_url
    if is_nova:
        # Nova.astrometry.net uses numeric IDs and different URL format
        wcs_fits_link = "https://nova.astrometry.net/wcs_file/{:d}".format(solved_id)
    else:
        # Custom server - use jobs/<id>/wcs_file endpoint with UUID
        wcs_fits_link = api_url.rstrip('/') + "/jobs/{}/wcs_file".format(solved_id)
    wcs_fits = urlopen(wcs_fits_link).read()

    # Load the WCS file
    # Suppress warnings about standalone WCS (no image data) and non-standard SIP keywords
    import logging
    astropy_logger = logging.getLogger('astropy')
    original_level = astropy_logger.level
    astropy_logger.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FITSFixedWarning)
        warnings.simplefilter('ignore', AstropyWarning)
        wcs_obj = WCS(fits.Header.fromstring(wcs_fits), naxis=2)
    astropy_logger.setLevel(original_level)

    # Print the WCS fields
    print("WCS fields:")
    print(wcs_obj)

    print("Astrometry.net link:")
    _printWebLink(stat, first_status=first_status)
    print("Astrometry.net solution:")
    print(result)

    # Use image dimensions if available; otherwise, fallback on median star positions
    if (x_center is None) or (y_center is None):
        
        if img is not None:
            x_center = img.shape[1]/2.0
            y_center = img.shape[0]/2.0
        else:
            x_center = np.median(x_data)
            y_center = np.median(y_data)


    # Use the WCS file to compute the center and orientation
    ra_mid, dec_mid = wcs_obj.all_pix2world(x_center, y_center, 1)

    # Image coordinate slightly right of the centre
    x_right = x_center + 10
    y_right = y_center
    ra_right, dec_right = wcs_obj.all_pix2world(x_right, y_right, 1)

    # Compute the equatorial orientation
    rot_eq_standard = np.degrees(np.arctan2(np.radians(dec_mid) - np.radians(dec_right), \
            np.radians(ra_mid) - np.radians(ra_right)))%360

    # Compute the scale from server response or WCS
    if result.get('pixscale', 0) > 0:
        scale = 3600/result['pixscale']
    else:
        # Compute scale from WCS pixel_scale_matrix
        # This gives the transformation matrix in deg/pixel
        psm = wcs_obj.pixel_scale_matrix
        # Pixel scale in deg/pixel from matrix determinant
        pixscale_deg = np.sqrt(np.abs(np.linalg.det(psm)))
        if pixscale_deg > 0:
            scale = 1.0 / pixscale_deg  # px/deg
        else:
            scale = 100.0  # fallback

    # Compute the FOV width and height from server response or estimate from WCS
    if result.get('width_arcsec', 0) > 0 and result.get('height_arcsec', 0) > 0:
        fov_w = result['width_arcsec']/3600
        fov_h = result['height_arcsec']/3600
    else:
        # Estimate FOV from image size and scale
        # Use NAXIS from WCS or estimate from center coordinates
        wcs_header = wcs_obj.to_header()
        naxis1 = wcs_header.get('IMAGEW', wcs_header.get('NAXIS1', 2*x_center))
        naxis2 = wcs_header.get('IMAGEH', wcs_header.get('NAXIS2', 2*y_center))
        fov_w = naxis1 / scale  # degrees
        fov_h = naxis2 / scale  # degrees

    # clean up temp image
    if tmpimg:
        try:
            os.remove(tmpimg)
        except Exception as e:
            sys.stderr.write("Warning: failed to remove temporary image '{}': {}\n".format(tmpimg, e))

    # Try to fetch matched star data from server (for custom servers that support it)
    matched_stars = []
    solution_info = None
    if api_url is not None and 'nova.astrometry.net' not in api_url:
        try:
            info_url = api_url.rstrip('/') + "/jobs/{}/info".format(solved_id)
            info_resp = urlopen(info_url)
            info_data = json.loads(info_resp.read())

            # Parse star_data if available
            if info_data.get('star_data'):
                for star in info_data['star_data']:
                    matched_stars.append({
                        'ra_deg': star.get('ra', 0),
                        'dec_deg': star.get('dec', 0),
                        'x_pix': star.get('x', 0),
                        'y_pix': star.get('y', 0)
                    })
                print("Matched stars from server: {:d}".format(len(matched_stars)))

            # Build solution_info similar to local solver
            # SkyFit2 looks for 'quad_stars' for magenta boxes
            solution_info = {
                'quad_stars': matched_stars,  # Used by SkyFit2 for magenta markers
                'matched_stars': matched_stars,  # Keep for compatibility
                'wcs_obj': wcs_obj,
                'solve_time': info_data.get('solve_time', 0),
                'objects_in_field': info_data.get('objects_in_field', [])
            }
        except Exception as e:
            print("Could not fetch matched stars: {}".format(e))
            solution_info = {'wcs_obj': wcs_obj}

    # Return star_data as [x_coords, y_coords] for compatibility
    star_data = None
    if matched_stars:
        star_x = [s['x_pix'] for s in matched_stars]
        star_y = [s['y_pix'] for s in matched_stars]
        star_data = [np.array(star_x), np.array(star_y)]

    return ra_mid, dec_mid, rot_eq_standard, scale, fov_w, fov_h, star_data, solution_info


if __name__ == '__main__':

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Solve the FF file by uploading it to nova.astrometry.net.")

    arg_parser.add_argument('file_path', nargs=1, metavar='FILE_PATH', type=str, \
        help='Path to the FF file which will be read and uploaded to nova.astrometry.net.')

    # arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
    #     help="Path to a config file which will be used instead of the default one.")

    # arg_parser.add_argument('-d', '--distortion', action="store_true", \
    #     help="""Refine the distortion parameters.""")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Test data
    x_data = [ 317.27, 299.90, 679.93,1232.75,1214.72, 424.84, 336.80, 618.96,  82.06, 593.16, 1007.06, \
        1226.50,  68.94, 191.38, 456.25, 845.41,1173.48,1060.64, 226.13, 875.91, 334.52,   9.16, 533.94, \
        282.33,1122.07, 611.01, 631.82]

    y_data = [ 32.78, 118.32, 322.71, 531.85, 483.43, 456.22, 195.02,  80.52, 621.05, 575.53, 245.44, \
        662.36,  92.83, 146.83, 381.14, 340.89,  34.08,   8.75, 323.30, 679.99, 692.43, 448.28,  47.88, \
        621.21, 562.74, 126.25, 372.89]

    # Contact nova.astrometry.net for the solution
    print(novaAstrometryNetSolve(ff_file_path=cml_args.file_path[0]))#, x_data, y_data))
    
    # Returns:
    # {'parity': 1.0, 
    #     'width_arcsec': 186857.80594735427, 
    #     'ra': 351.38664749562014, 
    #     'pixscale': 151.54728787295562, 
    #     'radius': 29.77070194214109, 
    #     'dec': 75.03531583306331, 
    #     'height_arcsec': 105022.27049595825, 
    #     'orientation': -75.0809053003417}