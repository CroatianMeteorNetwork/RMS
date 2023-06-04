# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from dateutil import parser
import urllib.request
from html.parser import HTMLParser
import os
import datetime
import time
from ftplib import FTP

class HTMLFilter(HTMLParser):
    text = ""
    def handle_data(self, data):
        self.text += data


tmpdirectory = "/tmp/"

def construct(ts):



        """ Turn an event into a string for easy comprehension


        Arguments:


        Return:
            String representation of an event
        """

        print("Start Lat {:3.5f}, Lon {:3.5f}, ht {:3.0f}km".format(float(ts[63]), float(ts[65]), float(ts[67])))
        print("End   Lat {:3.5f}, Lon {:3.5f}, ht {:3.0f}km".format(float(ts[69]), float(ts[71]), float(ts[73])))


        output = "# Required \n"
        output += ("EventTime                : {}\n".format(parser.parse(ts[2]).strftime("%Y%m%d_%H%M%S")))
        output += ("TimeTolerance (s)        : {:.0f}\n".format(60))
        output += ("EventLat (deg +N)        : {:3.5f}\n".format(float(ts[63])))
        #output += ("EventLatStd (deg)        : {:3.2f}\n".format(self.latstd))
        output += ("EventLon (deg +E)        : {:3.5f}\n".format(float(ts[65])))
        #output += ("EventLonStd (deg)        : {:3.2f}\n".format(self.lonstd))
        output += ("EventHt (km)             : {:3.5f}\n".format(float(ts[67])))
        #output += ("EventHtStd (km)          : {:3.2f}\n".format(self.htstd))
        output += ("CloseRadius(km)          : {:3.2f}\n".format(10))
        output += ("FarRadius (km)           : {:3.2f}\n".format(600))
        #output += ("\n")
        #output += ("# Optional second point      \n")
        output += ("EventLat2 (deg +N)       : {:3.5f}\n".format(float(ts[69])))
        #output += ("EventLat2Std (deg)       : {:3.2f}\n".format(self.lat2std))
        output += ("EventLon2 (deg +E)       : {:3.5f}\n".format(float(ts[71])))
        #output += ("EventLon2Std (deg)       : {:3.2f}\n".format(float(ts[73])))
        output += ("EventHt2 (km)            : {:3.5f}\n".format(float(ts[73])))
        #output += ("EventHtStd2 (km)         : {:3.2f}\n".format(self.ht2std))
        #output += ("\n")
        #output += ("# Or a trajectory instead    \n")
        #output += ("EventAzim (deg +E of N)  : {:3.2f}\n".format(self.azim))
        #output += ("EventAzimStd (deg)       : {:3.2f}\n".format(self.azimstd))
        #output += ("EventElev (deg)          : {:3.2f}\n".format(self.elev))
        #output += ("EventElevStd (deg):      : {:3.2f}\n".format(self.elevstd))
        #output += ("EventElevIsMax           : {:3.2f}\n".format(self.elevismax))
        #output += ("\n")
        #output += ("# Control information        \n")
        #output += ("StationsRequired         : {}\n".format(self.stationsrequired))
        #output += ("uuid                     : {}\n".format(self.uuid))
        output += ("RespondTo                : {}\n".format("192.168.1.241"))



        output += ("\n")
        output += ("END")
        output += ("\n")
        output += ("\n")
        output += ("\n")

        return output





def gettrajectoriesfromgmn(etime,tolerance):


    f = HTMLFilter()
    eventdate = etime[0:8]
    eventtime = etime[9:15]
    dt = datetime.datetime.strptime(etime.strip(), "%Y%m%d_%H%M%S")
    data = urllib.request.urlopen("https://globalmeteornetwork.org/data/traj_summary_data/daily/").read().decode("utf-8")
    f.feed(data)
    candidatetrajectories = []
    for line in f.text.splitlines():


        try:
         linetime = datetime.datetime.strptime(line[13:21], "%Y%m%d")
        except:
         continue


        if 0 > (((linetime - dt).total_seconds())/3600/24) > -2   :

            filetodownload = line.split(" ")[0]

            trajectories = urllib.request.urlopen(os.path.join("https://globalmeteornetwork.org/data/traj_summary_data/daily/",filetodownload)).read().decode("utf-8").splitlines()
            for trajectory in trajectories:

                trajectorysplit = trajectory.split(';')

                if trajectorysplit[0] != "" and trajectorysplit[0][0] != "#":

                    trajectorytime = parser.parse(trajectorysplit[2])

                    if abs((trajectorytime-dt).total_seconds()) < int(tolerance):
                        candidatetrajectories.append(trajectorysplit)
    return candidatetrajectories


def addeventtoFTP(eventinstruction):

    f = open(os.path.join(tmpdirectory, "event_watchlist.txt"), "wb")

    with FTP("192.168.1.241") as ftp:
        ftp.login('anonymous')
        ftp.cwd('data')
        try:
         ftp.retrbinary('RETR event_watchlist.txt',f.write)
        except:
         pass
        f.close

    f = open(os.path.join(tmpdirectory,"event_watchlist.txt"), "a")
    f.write(eventinstruction)
    f.close
    with FTP("192.168.1.241") as ftp:
        f = open(os.path.join(tmpdirectory,"event_watchlist.txt"), "rb")
        ftp.login('anonymous')
        ftp.cwd('data')
        ftp.storlines("STOR " + "event_watchlist.txt", f)
        f.close



def search(event_time,timetolerance):

    candidatetrajectories = gettrajectoriesfromgmn(event_time, timetolerance)
    print("Candidate Trajectories {}".format(candidatetrajectories))
    print("Following candidate trajectories were identified")
    trajectorycounter = 1
    for traj in candidatetrajectories:


        print("Trajectory {}".format(trajectorycounter))
        print("Date {}, Stations".format(traj[2]), traj[85].strip())
        print("Start Lat {:3.5f}, Lon {:3.5f}, ht {:3.0f}km".format(float(traj[63]), float(traj[65]),
                                                                    float(traj[67])))
        print("End   Lat {:3.5f}, Lon {:3.5f}, ht {:3.0f}km".format(float(traj[69]), float(traj[71]),
                                                                    float(traj[73])))
        trajectorycounter += 1
    print("Select trajectory")
    trajectorynumber = int(input())
    print("Constructing EventMonitor instruction for trajectory {}".format(trajectorynumber))
    print(candidatetrajectories[trajectorynumber - 1])
    instruction = construct(candidatetrajectories[trajectorynumber - 1])

    return instruction

def manualinput(event_time, timetolerance):

    print("Start Latitude")
    lat1 = input()
    print("Start Lon")
    lon1 = input()
    print("Start height (km)")
    ht1 = input()

    print("Do you wish to enter and end point (y/n)")
    enterend = input().upper()
    if enterend == "Y":
     print("End Latitude")
     lat2 = input()
     print("End Lon")
     lon2 = input()
     print("End height (km)")
     ht2 = input()
    else:
     print("Handling as a single point event")
     lat2, lon2, ht2 = lat1, lon1, ht1
    print("Enter inner radius in which all cameras must respond km")
    close_radius = input()
    print("Enter outer radius in which only cameras with suitable FoV should respond")
    far_radius = input()



    output = "# Required \n"
    output += ("EventTime                : {}\n".format(event_time))
    output += ("TimeTolerance (s)        : {:.0f}\n".format(float(timetolerance)))
    output += ("EventLat (deg +N)        : {:3.5f}\n".format(float(lat1)))
    # output += ("EventLatStd (deg)        : {:3.2f}\n".format(self.latstd))
    output += ("EventLon (deg +E)        : {:3.5f}\n".format(float(lon1)))
    # output += ("EventLonStd (deg)        : {:3.2f}\n".format(self.lonstd))
    output += ("EventHt (km)             : {:3.5f}\n".format(float(ht1)))
    # output += ("EventHtStd (km)          : {:3.2f}\n".format(self.htstd))
    output += ("CloseRadius(km)          : {:3.2f}\n".format(float(close_radius)))
    output += ("FarRadius (km)           : {:3.2f}\n".format(float(far_radius)))
    # output += ("\n")
    # output += ("# Optional second point      \n")
    output += ("EventLat2 (deg +N)       : {:3.5f}\n".format(float(lat2)))
    # output += ("EventLat2Std (deg)       : {:3.2f}\n".format(self.lat2std))
    output += ("EventLon2 (deg +E)       : {:3.5f}\n".format(float(lon2)))
    # output += ("EventLon2Std (deg)       : {:3.2f}\n".format(float(ts[73])))
    output += ("EventHt2 (km)            : {:3.5f}\n".format(float(ht2)))
    # output += ("EventHtStd2 (km)         : {:3.2f}\n".format(self.ht2std))
    # output += ("\n")
    # output += ("# Or a trajectory instead    \n")
    # output += ("EventAzim (deg +E of N)  : {:3.2f}\n".format(self.azim))
    # output += ("EventAzimStd (deg)       : {:3.2f}\n".format(self.azimstd))
    # output += ("EventElev (deg)          : {:3.2f}\n".format(self.elev))
    # output += ("EventElevStd (deg):      : {:3.2f}\n".format(self.elevstd))
    # output += ("EventElevIsMax           : {:3.2f}\n".format(self.elevismax))
    # output += ("\n")
    # output += ("# Control information        \n")
    # output += ("StationsRequired         : {}\n".format(self.stationsrequired))
    # output += ("uuid                     : {}\n".format(self.uuid))
    output += ("RespondTo                : {}\n".format("192.168.1.241"))

    output += ("\n")
    output += ("END")
    output += ("\n")
    output += ("\n")
    output += ("\n")

    return output


def construct(ts):
    """ Turn an event into a string for easy comprehension


    Arguments:


    Return:
        String representation of an event
    """

    print("Start Lat {:3.5f}, Lon {:3.5f}, ht {:3.0f}km".format(float(ts[63]), float(ts[65]), float(ts[67])))
    print("End   Lat {:3.5f}, Lon {:3.5f}, ht {:3.0f}km".format(float(ts[69]), float(ts[71]), float(ts[73])))

    output = "# Required \n"
    output += ("EventTime                : {}\n".format(parser.parse(ts[2]).strftime("%Y%m%d_%H%M%S")))
    output += ("TimeTolerance (s)        : {:.0f}\n".format(60))
    output += ("EventLat (deg +N)        : {:3.5f}\n".format(float(ts[63])))
    # output += ("EventLatStd (deg)        : {:3.2f}\n".format(self.latstd))
    output += ("EventLon (deg +E)        : {:3.5f}\n".format(float(ts[65])))
    # output += ("EventLonStd (deg)        : {:3.2f}\n".format(self.lonstd))
    output += ("EventHt (km)             : {:3.5f}\n".format(float(ts[67])))
    # output += ("EventHtStd (km)          : {:3.2f}\n".format(self.htstd))
    output += ("CloseRadius(km)          : {:3.2f}\n".format(10))
    output += ("FarRadius (km)           : {:3.2f}\n".format(600))
    # output += ("\n")
    # output += ("# Optional second point      \n")
    output += ("EventLat2 (deg +N)       : {:3.5f}\n".format(float(ts[69])))
    # output += ("EventLat2Std (deg)       : {:3.2f}\n".format(self.lat2std))
    output += ("EventLon2 (deg +E)       : {:3.5f}\n".format(float(ts[71])))
    # output += ("EventLon2Std (deg)       : {:3.2f}\n".format(float(ts[73])))
    output += ("EventHt2 (km)            : {:3.5f}\n".format(float(ts[73])))
    # output += ("EventHtStd2 (km)         : {:3.2f}\n".format(self.ht2std))
    # output += ("\n")
    # output += ("# Or a trajectory instead    \n")
    # output += ("EventAzim (deg +E of N)  : {:3.2f}\n".format(self.azim))
    # output += ("EventAzimStd (deg)       : {:3.2f}\n".format(self.azimstd))
    # output += ("EventElev (deg)          : {:3.2f}\n".format(self.elev))
    # output += ("EventElevStd (deg):      : {:3.2f}\n".format(self.elevstd))
    # output += ("EventElevIsMax           : {:3.2f}\n".format(self.elevismax))
    # output += ("\n")
    # output += ("# Control information        \n")
    # output += ("StationsRequired         : {}\n".format(self.stationsrequired))
    # output += ("uuid                     : {}\n".format(self.uuid))
    output += ("RespondTo                : {}\n".format("192.168.1.241"))

    output += ("\n")
    output += ("END")
    output += ("\n")
    output += ("\n")
    output += ("\n")

    return output


def createinstruction():

    print("Enter event time YYYYMMDD_HHMMSS")
    event_time = input()
    print("Enter time tolerance (s)")
    timetolerance = input()
    print("Search for event in GMN database (y/n) ?")
    searchindatabase = input()
    if searchindatabase.upper() == "Y":
        return search(event_time, timetolerance)

    else:
        return manualinput(event_time, timetolerance)

def interactiveinput():



    print("Remove Event / Add Event / Quit (r/a/q)")
    manageorcreate = input().upper()
    if manageorcreate == "A":
       instruction = createinstruction()
       print("Following instruction has been prepared.")
       print("")
       print(instruction)
       print("Submit (y/n)")
       submit = input().upper()
       print("submitted {}".format(submit.upper()))
       if submit == "Y":
           addeventtoFTP(instruction)

    if manageorcreate == "Q":
        quit()

if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="""Simple UI for Eventmonitor. \
        """, formatter_class=argparse.RawTextHelpFormatter)



    arg_parser.add_argument('-i', '--interactive', dest='interactive', default=False, action="store_true", \
                    help="Run in interactive mode.")



    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    if cml_args.interactive:
        interactiveinput()