
from dateutil import parser
import urllib.request
from html.parser import HTMLParser
import os
import datetime
import time
from ftplib import FTP
import glob
import RMS.ConfigReader as cr
from geopy.geocoders import Nominatim


webroot = "/home/em/ftp/data/events/"
stationdata = "stationdata"

class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data):
        self.text += data

def reversegeo(lat, lon, ID):

        print(ID)

        if ID == "AU0006" or ID == "AU0009":
            network = "WA"
            station = "Lemon"

        if ID == "AU000U" or ID == "AU000V" or ID == "AU000W" or ID == "AU000X" or ID == "AU000Y" or ID == "AU000Z":
            network = "WA"
            station = "Pioneer"

        if ID == "AU000A" or ID == "AU000C" or ID == "AU000D" or ID == "AU000E" or ID == "AU000F" or ID == "AU000G" or ID == "AU000H" or ID == "AU000K":
            network = "WA"
            station = "Walnut"

        if ID == "AU001A" or ID == "AU001B" or ID == "AU001C" or ID == "AU001D" or ID == "AU001E" or ID == "AU001F" :
            network = "WA"
            station = "Rhodesdale"

        if ID == "AU001P" or ID == "AU001Q" or ID == "AU001R" or ID == "AU001S" or ID == "AU001T":
            network = "WA"
            station = "Lemon"

        if ID == "AU001U" or ID == "AU001V" or ID == "AU001W" or ID == "AU001X" or ID == "AU001Y" or ID == "AU001Z" :
            network = "WA"
            station = "Coorinja"


        return network, station


def server():





    for camera in glob.glob("/home/au*"):

        incomingdir = os.path.join(camera,"event_monitor")
        if os.path.exists(incomingdir):

            files = os.listdir(incomingdir)

            for bz2file in files:
                if bz2file[0] == ".":
                    continue
                eventdir = bz2file[7:22]

                shellcommand = ""
                shellcommand += "mkdir -p {} ; \n".format(os.path.join(webroot,eventdir,stationdata))
                shellcommand += "mkdir -p {} ; \n".format(os.path.join(webroot,eventdir, "summary"))
                shellcommand += "mkdir -p {} ; \n".format(os.path.join(webroot, eventdir, "_videosandimages"))
                shellcommand += "cp {} {}; \n".format(os.path.join(camera,"event_monitor",bz2file),os.path.join(webroot,eventdir,stationdata))
                shellcommand += "cd {} ; ".format(os.path.join(webroot, eventdir, stationdata))
                shellcommand += "tar -xf {} ;".format(os.path.join(webroot,eventdir,stationdata,bz2file))
                #shellcommand += "rm {} ;".format(os.path.join(webroot, eventdir, stationdata, bz2file))
                shellcommand += "cd {} ; ".format(os.path.join(webroot, eventdir, "summary"))


                print(shellcommand)
                os.system(shellcommand)
                print(os.path.join(webroot, eventdir, stationdata,bz2file[0:22],".config"))
                config = cr.parse(os.path.join(webroot, eventdir, stationdata,bz2file[0:22],".config"))

                print("lat {} and lon {}".format(config.latitude,config.longitude ))

                network,station = reversegeo(config.latitude,config.longitude, config.stationID)
                print("Network {} Station {}".format(network,station))
                shellcommand = ""
                stationpath = os.path.join(os.path.join(webroot, eventdir, "summary",network,station))

                camerapath = os.path.join(stationpath, os.path.basename(camera).upper())
                shellcommand += "mkdir -p {} ; \n".format(camerapath)
                shellcommand += "cd {} ; \n".format(camerapath)
                shellcommand += "cp {}/*.fits . ;\n".format(os.path.join(webroot, eventdir, stationdata, bz2file[0:22]))
                shellcommand += "cp {}/*.jpg  . ;\n".format(os.path.join(webroot, eventdir, stationdata, bz2file[0:22]))
                shellcommand += "cp {}/*.jpg  . ;\n".format(os.path.join(webroot, eventdir, stationdata, bz2file[0:22]))
                shellcommand += "cp {}/*.bin  . ;\n".format(os.path.join(webroot, eventdir, stationdata, bz2file[0:22]))
                shellcommand += "cp {}/*.mp4  . ;\n".format(os.path.join(webroot, eventdir, stationdata, bz2file[0:22]))
                shellcommand += "for f in *.mp4 ;       do mv $f ../../../../_videosandimages/" + station + "_${f##*/} ; done ; \n"
                #shellcommand += "for f in *mosaic.png ; do mv $f ../../_videosandimages/" + station + "_${f##*/} ; done"
                shellcommand += "mv " + camerapath + "/*_stack.jpg " + "../../../../_videosandimages/" + station + "_" + os.path.basename(camera).upper() + "_stack.jpg ; \n"
                shellcommand += "cd {} ;".format(webroot)
                shellcommand += "mkdir -p bz2files ; \n"
                #shellcommand += "tar -cjf bz2files/" + eventdir + ".tar.bz2 " + eventdir
                print(shellcommand)
                os.system(shellcommand)


if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Simple UI for Eventmonitor. \
        """, formatter_class=argparse.RawTextHelpFormatter)






    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    server()

            



    
