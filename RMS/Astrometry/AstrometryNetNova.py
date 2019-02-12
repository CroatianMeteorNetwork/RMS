#!/usr/bin/env python

""" 
Astrometry.net client script for communicating with astrometry.net servers.
Modified from: https://github.com/dstndstn/astrometry.net/blob/master/net/client/client.py
"""

from __future__ import print_function

import os
import sys
import time
import base64

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

#from exceptions import Exception
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.application  import MIMEApplication

from email.encoders import encode_noop

import json


# Denis' API key for nova.astrometry.net
API_KEY = "sybwjtfjbrpgomep"


DEBUG = False


def printDebug(*args):
    if DEBUG:
        printDebug(*args)


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
            # Make a custom generator to format it the way we need.
            from io import BytesIO
            try:
                # py3
                from email.generator import BytesGenerator as TheGenerator
            except ImportError:
                # py2
                from email.generator import Generator as TheGenerator

            m1 = MIMEBase('text', 'plain')
            m1.add_header('Content-disposition',
                          'form-data; name="request-json"')
            m1.set_payload(json)
            m2 = MIMEApplication(file_args[1],'octet-stream',encode_noop)
            m2.add_header('Content-disposition',
                          'form-data; name="file"; filename="%s"'%file_args[0])
            mp = MIMEMultipart('form-data', None, [m1, m2])

            class MyGenerator(TheGenerator):
                def __init__(self, fp, root=True):
                    # don't try to use super() here; in py2 Generator is not a
                    # new-style class.  Yuck.
                    TheGenerator.__init__(self, fp, mangle_from_=False,
                                          maxheaderlen=0)
                    self.root = root
                def _write_headers(self, msg):
                    # We don't want to write the top-level headers;
                    # they go into Request(headers) instead.
                    if self.root:
                        return
                    # We need to use \r\n line-terminator, but Generator
                    # doesn't provide the flexibility to override, so we
                    # have to copy-n-paste-n-modify.
                    for h, v in msg.items():
                        self._fp.write(('%s: %s\r\n' % (h,v)).encode())
                    # A blank line always separates headers from body
                    self._fp.write('\r\n'.encode())

                # The _write_multipart method calls "clone" for the
                # subparts.  We hijack that, setting root=False
                def clone(self, fp):
                    return MyGenerator(fp, root=False)

            fp = BytesIO()
            g = MyGenerator(fp)
            g.flatten(mp)
            data = fp.getvalue()
            headers = {'Content-type': mp.get('Content-type')}

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
        sess = result.get('session')
        printDebug('Got session:', sess)
        if not sess:
            raise RequestError('no session in result')
        self.session = sess

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
            # image_width, image_height
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

    def upload(self, fn=None, **kwargs):
        args = self._get_upload_args(**kwargs)
        file_args = None
        if fn is not None:
            try:
                f = open(fn, 'rb')
                file_args = (fn, f.read())
            except IOError:
                printDebug('File %s does not exist' % fn)
                raise
        return self.send_request('upload', args, file_args)

    def submission_images(self, subid):
        result = self.send_request('submission_images', {'subid':subid})
        return result.get('image_ids')

    def overlay_plot(self, service, outfn, wcsfn, wcsext=0):
        from astrometry.util import util as anutil
        wcs = anutil.Tan(wcsfn, wcsext)
        params = dict(crval1 = wcs.crval[0], crval2 = wcs.crval[1],
                      crpix1 = wcs.crpix[0], crpix2 = wcs.crpix[1],
                      cd11 = wcs.cd[0], cd12 = wcs.cd[1],
                      cd21 = wcs.cd[2], cd22 = wcs.cd[3],
                      imagew = wcs.imagew, imageh = wcs.imageh)
        result = self.send_request(service, {'wcs':params})
        printDebug('Result status:', result['status'])
        plotdata = result['plot']
        plotdata = base64.b64decode(plotdata)
        open(outfn, 'wb').write(plotdata)
        printDebug('Wrote', outfn)

    def sdss_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('sdss_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def galex_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('galex_image_for_wcs', outfn,
                                 wcsfn, wcsext)

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

    def annotate_data(self,job_id):
        """
        :param job_id: id of job
        :return: return data for annotations
        """
        result = self.send_request('jobs/%s/annotations' % job_id)
        return result

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



def novaAstrometryNetSolveXY(x_data, y_data, api_key=None):
    """ Find an astrometric solution of X, Y image coordinates of stars detected on an image using the 
        nova.astrometry.net service.

    Arguments:
        x_data: [list] A list of star x image coordiantes.
        y_data: [list] A list of star y image coordiantes.

    Keyword arguments:
        api_key: [str] nova.astrometry.net user API key. None by default, in which case the default API
            key will be used.

    Return:
        (ra, dec, orientation, scale, fov_w, fov_h): [tuple of floats] All in degrees, scale in px/deg.
    """

    c = Client()

    # Log in to nova.astrometry.net
    if api_key is None:
        api_key = API_KEY

    c.login(api_key)

    # Upload the list of stars
    upres = c.upload(x=x_data, y=y_data)

    stat = upres['status']
    if stat != 'success':
        
        print('Upload failed: status', stat)
        print(upres)

        return False

    # Submission ID
    sub_id = upres['subid']

    # Wait until the plate is solved
    while True:
        
        stat = c.sub_status(sub_id, justdict=True)
        print('Got status:', stat)
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

    # Get results
    while True:
        stat = c.job_status(solved_id, justdict=True)

        # Get the calibration
        result = c.send_request('jobs/%s/calibration' % solved_id)
        print('Got job status:', stat)
        if stat.get('status','') in ['success']:
            print(result)
            break

        time.sleep(5)


    # RA/Dec of centre
    ra = result['ra']
    dec = result['dec']

    # Orientation +E of N
    orientation = result['orientation']

    # Image scale in px/deg
    scale = 3600/result['pixscale']

    # FOV in deg
    fov_w = result['width_arcsec']/3600
    fov_h = result['height_arcsec']/3600

    return ra, dec, orientation, scale, fov_w, fov_h


if __name__ == '__main__':

    # Test data
    x_data = [ 317.27, 299.90, 679.93,1232.75,1214.72, 424.84, 336.80, 618.96,  82.06, 593.16, 1007.06, \
        1226.50,  68.94, 191.38, 456.25, 845.41,1173.48,1060.64, 226.13, 875.91, 334.52,   9.16, 533.94, \
        282.33,1122.07, 611.01, 631.82]

    y_data = [ 32.78, 118.32, 322.71, 531.85, 483.43, 456.22, 195.02,  80.52, 621.05, 575.53, 245.44, \
        662.36,  92.83, 146.83, 381.14, 340.89,  34.08,   8.75, 323.30, 679.99, 692.43, 448.28,  47.88, \
        621.21, 562.74, 126.25, 372.89]

    # Contact nova.astrometry.net for the solution
    novaAstrometryNetSolveXY(x_data, y_data)
    
    # Returns:
    # {'parity': 1.0, 
    #     'width_arcsec': 186857.80594735427, 
    #     'ra': 351.38664749562014, 
    #     'pixscale': 151.54728787295562, 
    #     'radius': 29.77070194214109, 
    #     'dec': 75.03531583306331, 
    #     'height_arcsec': 105022.27049595825, 
    #     'orientation': -75.0809053003417}