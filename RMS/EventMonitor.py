# RPi Meteor Station
# Copyright (C) 2016
#
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



""""" Automatically uploads data files based on time and trajectory information given on a website. """

from __future__ import print_function, division, absolute_import

import sqlite3
import multiprocessing
import RMS.ConfigReader as cr
import urllib.request
import os
import shutil
import time
import copy
import uuid
import numpy as np
import datetime
import argparse
import math
import random, string
from glob import glob


from RMS.Astrometry.Conversions import datetime2JD, geo2Cartesian, altAz2RADec, latLonAlt2ECEF, vectNorm
from RMS.Math import angularSeparationVect
from RMS.Formats.Platepar import Platepar
from datetime import datetime
from dateutil import parser
from Utils.StackFFs import stackFFs
from Utils.BatchFFtoImage import batchFFtoImage
from RMS.Astrometry.CyFunctions import cyTrueRaDec2ApparentAltAz
from RMS.UploadManager import uploadSFTP
import logging


log = logging.getLogger("logger")


class EventContainer(object):

    def __init__(self, dt, lat, lon, ht):

        # Required parameters

        self.dt,self.timetolerance = dt, 0

        self.lat, self.latstd, self.lon, self.lonstd, self.ht, self.htstd = lat, 0, lon, 0, ht,0
        self.lat2, self.lat2std, self.lon2, self.lon2std, self.ht2, self.ht2std = 0,0,0,0,0,0
        self.closeradius, self.farradius = 0,0

        # Or trajectory information from the first point
        self.azim, self.azimstd, self.elev, self.elevstd, self.elevismax = 0,0,0,0,False

        self.stationsrequired = ""
        self.respondto = ""

        # These are internal control properties

        self.uuid = ""
        self.eventspectype = 0
        self.filesuploaded = []
        self.timecompleted = None
        self.observedstatus = None
        self.processedstatus = False

        self.startdistance, self.startangle, self.enddistance, self.endangle = 0,0,0,0
        self.fovra, self.fovdec = 0 , 0


    def setvalue(self, variable_name, value):

        """ Receive a name and value pair, and put them into this event

        Arguments:
            variable_name: Name of the variable
            value        : Value to be assigned

        Return:
            Nothing
        """
        # Extract the variable name, truncate before any '(' used for units
        variable_name = variable_name.strip().split('(')[0].strip()

        if value == "":
            return

        # Mandatory parameters

        self.dt = value if "EventTime" == variable_name else self.dt
        self.timetolerance = value if "TimeTolerance" == variable_name else self.timetolerance
        self.lat = float(value) if "EventLat" == variable_name else self.lat
        self.latstd = float(value) if "EventLatStd" == variable_name else self.latstd
        self.lon = float(value) if "EventLon" == variable_name else self.lon
        self.lonstd = float(value) if "EventLonStd" == variable_name else self.lonstd
        self.ht = float(value) if "EventHt" == variable_name else self.ht
        self.htstd = float(value) if "EventHtStd" == variable_name else self.htstd

        # radii

        self.closeradius = float(value) if "CloseRadius" == variable_name else self.closeradius
        self.farradius = float(value) if "FarRadius" == variable_name else self.farradius

        # Optional parameters, if trajectory is set by a start and an end

        self.lat2 = float(value) if "EventLat2" == variable_name else self.lat2
        self.lat2std = float(value) if "EventLat2Std" == variable_name else self.lat2std
        self.lon2 = float(value) if "EventLon2" == variable_name else self.lon2
        self.lon2std = float(value) if "EventLon2Std" == variable_name else self.lon2std
        self.ht2 = float(value) if "EventHt2" == variable_name else self.ht2
        self.ht2std = float(value) if "EventHt2Std" == variable_name else self.ht2std

        # Optional parameters for defining trajectory by a start point, and a direction

        if "EventAzim" == variable_name:
            if value is None:
                self.azim = 0
            else:
                self.azim = float(value)

        if "EventAzimStd" == variable_name:
            if value is None:
                self.azimstd = 0
            else:
                self.azimstd = float(value)

        if "EventElev" == variable_name:
            if value is None:
                self.elev = 0
            else:
                self.elev = float(value)

        if "EventElevStd" == variable_name:
            if value is None:
                self.elevstd = 0
            else:
                self.elevstd = float(value)

        if "EventElevIsMax" == variable_name:
            if value == "True":
                self.elevismax = True
            else:
                self.elevismax = False

        # Control information

        self.stationsrequired = str(value) if "StationsRequired" == variable_name else self.stationsrequired
        self.uuid = str(value) if "uuid" == variable_name else self.uuid
        self.respondto = str(value) if "RespondTo" == variable_name else self.respondto

    def eventtostring(self):

        """ Turn an event into a string

        Arguments:

        Return:
            String representation of an event
        """

        output = "# Required \n"
        output += ("EventTime                : {}\n".format(self.dt))
        output += ("TimeTolerance (s)        : {:.0f}\n".format(self.timetolerance))
        output += ("EventLat (deg +N)        : {:3.2f}\n".format(self.lat))
        output += ("EventLatStd (deg)        : {:3.2f}\n".format(self.latstd))
        output += ("EventLon (deg +E)        : {:3.2f}\n".format(self.lon))
        output += ("EventLonStd (deg)        : {:3.2f}\n".format(self.lonstd))
        output += ("EventHt (km)             : {:3.2f}\n".format(self.ht))
        output += ("EventHtStd (km)          : {:3.2f}\n".format(self.htstd))
        output += ("CloseRadius(km)          : {:3.2f}\n".format(self.closeradius))
        output += ("FarRadius (km)           : {:3.2f}\n".format(self.farradius))
        output += "\n"
        output += "# Optional second point      \n"
        output += ("EventLat2 (deg +N)       : {:3.2f}\n".format(self.lat2))
        output += ("EventLat2Std (deg)       : {:3.2f}\n".format(self.lat2std))
        output += ("EventLon2 (deg +E)       : {:3.2f}\n".format(self.lon2))
        output += ("EventLon2Std (deg)       : {:3.2f}\n".format(self.lon2std))
        output += ("EventHt2 (km)            : {:3.2f}\n".format(self.ht2))
        output += ("EventHtStd2 (km)         : {:3.2f}\n".format(self.ht2std))
        output += "\n"
        output += "# Or a trajectory instead    \n"
        output += ("EventAzim (deg +E of N)  : {:3.2f}\n".format(self.azim))
        output += ("EventAzimStd (deg)       : {:3.2f}\n".format(self.azimstd))
        output += ("EventElev (deg)          : {:3.2f}\n".format(self.elev))
        output += ("EventElevStd (deg):      : {:3.2f}\n".format(self.elevstd))
        output += ("EventElevIsMax           : {:3.2f}\n".format(self.elevismax))
        output += "\n"
        output += "# Control information        \n"
        output += ("StationsRequired         : {}\n".format(self.stationsrequired))
        output += ("uuid                     : {}\n".format(self.uuid))
        output += ("RespondTo                : {}\n".format(self.respondto))

        output += "# Trajectory information     \n"
        output += ("Start Distance (km)      : {:3.2f}\n".format(self.startdistance / 1000))
        output += ("Start Angle              : {:3.2f}\n".format(self.startangle))
        output += ("End Distance (km)        : {:3.2f}\n".format(self.enddistance / 1000))
        output += ("End Angle                : {:3.2f}\n".format(self.endangle))
        output += "# Station information        \n"
        output += ("Field of view RA         : {:3.2f}\n".format(self.fovra))
        output += ("Field of view Dec        : {:3.2f}\n".format(self.fovdec))

        output += "\n"
        output += "END"
        output += "\n"

        return output

    def checkreasonable(self):

        """ Receive an event, check if it is reasonable, and optionally try to fix it up
            Crucially, this function prevents any excessive requests being made that may compromise capture

        Arguments:


        Return:
            reasonable: [bool] The event is reasonable
        """


        reasonable = True

        reasonable = False if self.lat == "" else reasonable
        reasonable = False if self.lat is None else reasonable
        reasonable = False if self.lon == "" else reasonable
        reasonable = False if self.lon is None else reasonable
        reasonable = False if float(self.timetolerance) > 300 else reasonable
        reasonable = False if self.closeradius > self.farradius else reasonable

        return reasonable


class EventMonitor(multiprocessing.Process):

    def __init__(self, config):
        """ Automatically uploads data files of an event (e.g. fireball) as given on the website.
        Arguments:
            config: [Config] Configuration object.
        """



        super(EventMonitor, self).__init__()
        # Hold two configs - one for the locations of folders - syscon, and one for the lats and lons etc. - config
        self.config = config        #the config that will be used for all data processing - lats, lons etc.
        self.syscon = config        #the config that describes where the folders are
        # The path to the event monitor database
        self.event_monitor_db_path = os.path.join(os.path.abspath(self.config.data_dir),
                                                  self.config.event_monitor_db_name)
        self.conn = self.createEventMonitorDB()

        # Load the event monitor database. Any problems, delete and recreate.
        self.db_conn = self.getconnectiontoEventMonitorDB()
        self.exit = multiprocessing.Event()
        self.event_monitor_db_name = "event_monitor.db"

        self.check_interval = self.syscon.event_monitor_check_interval


        log.info("Started EventMonitor")
        log.info("Monitoring {} at {:3.2f} minute intervals".format(self.syscon.event_monitor_webpage,self.syscon.event_monitor_check_interval))
        log.info("Local db path name {}".format(self.syscon.event_monitor_db_name))
        log.info("Reporting data to {}/{}".format(self.syscon.hostname, self.syscon.event_monitor_remote_dir))


    def createEventMonitorDB(self, testmode = False):
        """ Creates the event monitor database. """

        # Create the event monitor database
        if testmode:
            self.event_monitor_db_path = os.path.expanduser(os.path.join(self.syscon.data_dir, self.event_monitor_db_name))
            if os.path.exists(self.event_monitor_db_path):
                os.unlink(self.event_monitor_db_path)

        if not os.path.exists(os.path.dirname(self.event_monitor_db_path)):
            # Handle the very rare case where this could run before any observation sessions
            # and RMS_data does not exist
            os.makedirs(os.path.dirname(self.event_monitor_db_path))

        conn = sqlite3.connect(self.event_monitor_db_path)
        log.info("Created a database at {}".format(self.event_monitor_db_path))
        tables = conn.cursor().execute(
            """SELECT name FROM sqlite_master WHERE type = 'table' and name = 'event_monitor';""").fetchall()

        if tables:
            return conn

        conn.execute("""CREATE TABLE event_monitor (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,   
                            EventTime TEXT NOT NULL,
                            TimeTolerance REAL NOT NULL,
                            EventLat REAL NOT NULL,
                            EventLatStd REAL NOT NULL,
                            EventLon REAL NOT NULL,
                            EventLonStd REAL NOT NULL,
                            EventHt REAL NOT NULL,
                            EventHtStd REAL NOT NULL,
                            CloseRadius REAL NOT NULL,
                            FarRadius REAL NOT NULL,
                            EventLat2 REAL NOT NULL,
                            EventLat2Std REAL NOT NULL,
                            EventLon2 REAL NOT NULL,
                            EventLon2Std REAL NOT NULL,
                            EventHt2 REAL NOT NULL,
                            EventHt2Std REAL NOT NULL,
                            EventAzim REAL NOT NULL,
                            EventAzimStd REAL NOT NULL,
                            EventElev REAL NOT NULL,
                            EventElevStd REAL NOT NULL,
                            EventElevIsMax BOOL,
                            filesuploaded TEXT,
                            timeadded TEXT,
                            timecompleted TEXT,
                            observedstatus BOOL,
                            processedstatus BOOL,
                            receivedbyserver BOOL,
                            uuid TEXT,              
                            RespondTo TEXT
                            )""")

        # Commit the changes
        conn.commit()

        # Close the connection
        self.db_conn = conn

    def delEventMonitorDB(self):

        """ Delete the event monitor database.


        Arguments:


        Return:
            Status: [bool] True if a db was found at that location, otherwise false

        """

        # This check is to prevent accidental deletion of the working directory

        if os.path.isfile(self.event_monitor_db_path):
            os.remove(self.event_monitor_db_path)
            return True
        else:
            return False

    def addDBcol(self, column, coltype):
        """ Add a new column to the database

        Arguments:
            column: [string] Name of column to add
            coltype: [string] type of columnd to add

        Return:
            Status: [bool] True if successful otherwise false

        """

        SQLCommand = ""
        SQLCommand += "ALTER TABLE event_monitor  \n"
        SQLCommand += "ADD {} {}; ".format(column, coltype)

        try:
            conn = sqlite3.connect(self.event_monitor_db_path)
            conn.execute(SQLCommand)
            conn.close()
            return True
        except:
            return False

    def deleteDBoldrecords(self):

        SQLCommand = ""
        SQLCommand += "DELETE FROM event_monitor \n "
        SQLCommand += "WHERE \n "
        SQLCommand += "timecompleted "
    def getconnectiontoEventMonitorDB(self):
        """ Loads the event monitor database

            Arguments:


            Return:
                connection: [connection] A connection to the database

            """

        # Create the event monitor database if it does not exist
        if not os.path.isfile(self.event_monitor_db_path):
            self.createEventMonitorDB()

        # Load the event monitor database - only gets done here
        try:
         self.conn = sqlite3.connect(self.event_monitor_db_path)
        except:
         os.unlink(self.event_monitor_db_path)
         self.createEventMonitorDB()

        return self.conn

    def eventexists(self, event):

        SQLStatement = ""
        SQLStatement += "SELECT COUNT(*) FROM event_monitor \n"
        SQLStatement += "WHERE \n"
        SQLStatement += "EventTime = '{}'          AND \n".format(event.dt)
        SQLStatement += "EventLat = '{}'               AND \n".format(event.lat)
        SQLStatement += "EventLon = '{}'               AND \n".format(event.lon)
        SQLStatement += "EventHt = '{}'                AND \n".format(event.ht)
        SQLStatement += "EventLatStd = '{}'            AND \n".format(event.latstd)
        SQLStatement += "EventLonStd = '{}'            AND \n".format(event.lonstd)
        SQLStatement += "EventHtStd = '{}'             AND \n".format(event.htstd)
        SQLStatement += "EventLat2 = '{}'              AND \n".format(event.lat2)
        SQLStatement += "EventLon2 = '{}'              AND \n".format(event.lon2)
        SQLStatement += "EventHt2 = '{}'               AND \n".format(event.ht2)
        SQLStatement += "EventLat2Std = '{}'           AND \n".format(event.lat2std)
        SQLStatement += "EventLon2Std = '{}'           AND \n".format(event.lon2std)
        SQLStatement += "EventHt2Std = '{}'            AND \n".format(event.ht2std)
        SQLStatement += "FarRadius = '{}'              AND \n".format(event.farradius)
        SQLStatement += "CloseRadius = '{}'            AND \n".format(event.closeradius)
        SQLStatement += "TimeTolerance = '{}'          AND \n".format(event.timetolerance)
        SQLStatement += "RespondTo = '{}'                  \n".format(event.respondto)

        # does a similar event exist
        # query gets the number of rows matching, not the actual rows

        try:
         return (self.db_conn.cursor().execute(SQLStatement).fetchone())[0] != 0
        except:
         log.info("Check for event exists failed")
         return False


    def deleteoldrecords(self):

        """

        Remove old record from the database, notional time of 14 days selected.
        The deletion is made on the criteria of when the record was added to the database, not the event date
        If the event is is still listed on the website, then it will be added, and uploaded.

        """



        SQLStatement = ""
        SQLStatement += "DELETE from event_monitor \n"
        SQLStatement += "WHERE                     \n"
        SQLStatement += "timeadded < date('now', '-14 day')"



        try:
            cursor = self.db_conn.cursor()
            cursor.execute(SQLStatement)
            self.db_conn.commit()

        except:
            log.info("Database purge failed")
            self.delEventMonitorDB()
            self.createEventMonitorDB()
        return None

    def addevent(self, event):

        """

        Checks to see if an event exists, if not then add to the database

            Arguments:
                event: [event] Event to be added to the database

            Return:
                added: [bool] True if added, else false

            """

        self.deleteoldrecords()

        if not self.eventexists(event):
            SQLStatement = ""
            SQLStatement += "INSERT INTO event_monitor \n"
            SQLStatement += "("
            SQLStatement += "EventTime, TimeTolerance,                   \n"
            SQLStatement += "EventLat ,EventLatStd ,EventLon , EventLonStd , EventHt ,EventHtStd,        \n"
            SQLStatement += "CloseRadius, FarRadius,                     \n"
            SQLStatement += "EventLat2, EventLat2Std, EventLon2, EventLon2Std,EventHt2, EventHt2Std,      \n"
            SQLStatement += "EventAzim, EventAzimStd, EventElev, EventElevStd, EventElevIsMax,    \n"
            SQLStatement += "processedstatus, uuid, RespondTo, timeadded \n"
            SQLStatement += ")                                           \n"

            SQLStatement += "VALUES "
            SQLStatement += "(                            \n"
            SQLStatement += "'{}',{},                     \n".format(event.dt, event.timetolerance)
            SQLStatement += "{},  {}, {}, {}, {}, {},     \n".format(event.lat, event.latstd, event.lon, event.lonstd,
                                                                     event.ht, event.htstd)
            SQLStatement += "{},  {},                     \n".format(event.closeradius, event.farradius)
            SQLStatement += "{},  {}, {}, {}, {}, {},     \n".format(event.lat2, event.lat2std, event.lon2,
                                                                     event.lon2std, event.ht2, event.ht2std)
            SQLStatement += "{},  {}, {}, {}, {} ,        \n".format(event.azim, event.azimstd, event.elev,
                                                                     event.elevstd,
                                                                     event.elevismax)
            SQLStatement += "{}, '{}', '{}',              \n".format(0, uuid.uuid4(), event.respondto)
            SQLStatement += "CURRENT_TIMESTAMP ) \n"

            try:
                cursor = self.db_conn.cursor()
                cursor.execute(SQLStatement)
                self.db_conn.commit()

            except:
                log.info("Add event failed")

            log.info("Added event at {} to the database".format(event.dt))
            return True
        else:
            #log.info("Event at {} already in the database".format(event.dt))
            return False

    def markeventasprocessed(self, event):

        """ Marks an event as having been processed

        Arguments:
            event: [event] Event to be marked as processed

        Return:
        """

        SQLStatement = ""
        SQLStatement += "UPDATE event_monitor                 \n"
        SQLStatement += "SET                                  \n"
        SQLStatement += "processedstatus = 1,                 \n"
        SQLStatement += "timecompleted   = CURRENT_TIMESTAMP  \n".format(datetime.now())
        SQLStatement += "                                     \n"
        SQLStatement += "WHERE                                \n"
        SQLStatement += "uuid = '{}'                          \n".format(event.uuid)
        try:
         self.db_conn.cursor().execute(SQLStatement)
         self.db_conn.commit()
         log.info("Event at {} marked as processed".format(event.dt))
        except:
         log.info("Database error")


    def markeventasuploaded(self, event, file_list):

        """ Checks to see if an event exists, if not then add to the database

            Arguments:
                event: [event] Event to be marked as uploaded
                file_list: [list of strings] Files uploaded

            Return:
        """

        files_uploaded = ""
        for file in file_list:
            files_uploaded += os.path.basename(file) + " "

        SQLStatement = ""
        SQLStatement += "UPDATE event_monitor \n"
        SQLStatement += "SET                  \n"
        SQLStatement += "filesuploaded  = '{}'\n".format(files_uploaded)
        SQLStatement += "                     \n"
        SQLStatement += "WHERE                \n"
        SQLStatement += "uuid = '{}'          \n".format(event.uuid)

        try:
         cursor = self.db_conn.cursor()
         cursor.execute(SQLStatement)
         self.db_conn.commit()
         log.info("Event at {} marked as uploaded".format(event.dt))
        except:
         log.info("Database error")

    def markeventasreceivedbyserver(self, uuid):

        """ Checks to see if an event exists, if not then add to the database

            Arguments:
                   uuid: [string] uuid of event received by server


            Return:
        """

        SQLStatement = ""
        SQLStatement += "UPDATE event_monitor     \n"
        SQLStatement += "SET                      \n"
        SQLStatement += "receivedbyserver =   '{}'\n".format("1")
        SQLStatement += "                         \n"
        SQLStatement += "WHERE                    \n"
        SQLStatement += "uuid = '{}'              \n".format(uuid)

        cursor = self.db_conn.cursor()
        cursor.execute(SQLStatement)
        self.db_conn.commit()

    def getEventsfromWebPage(self, testmode=False):

        """ Reads a webpage, and generates a list of events

            Arguments:

            Return:
                events : [list of events]
        """

        event = EventContainer(0, 0, 0, 0)  # Initialise it empty
        events = []

        if not testmode:
            try:
                web_page = urllib.request.urlopen(self.syscon.event_monitor_webpage).read().decode("utf-8").splitlines()

            except:
                # Return an empty list
                log.info("Event monitor found no page at {}".format(self.syscon.event_monitor_webpage))
                return events
        else:
            f = open(os.path.expanduser("~/RMS_data/event_watchlist.txt"), "r")
            web_page = f.read().splitlines()
            f.close()

        for line in web_page:

            line = line.split('#')[0]  # remove anything to the right of comments

            if ":" in line:  # then it is a value pair

                try:
                    variable_name = line.split(":")[0].strip()  # get variable name
                    value = line.split(":")[1].strip()  # get value
                    event.setvalue(variable_name, value)  # and put into this event container
                except:
                    pass

            else:
                if "END" in line:
                    events.append(copy.copy(event))  # copy, because these are references, not objects
                    event = EventContainer(0, 0, 0, 0)  # Initialise it empty
        log.info("Read {} events from {}".format(len(events), self.syscon.event_monitor_webpage))

        return events

    def getUnprocessedEventsfromDB(self):

        """ Get the unprocessed events from the database

            Arguments:

            Return:
                events : [list of events]
        """

        SQLStatement = ""
        SQLStatement += "SELECT "
        SQLStatement += ""
        SQLQueryCols = ""
        SQLQueryCols += "EventTime,TimeTolerance,EventLat,EventLatStd,EventLon, EventLonStd, EventHt, EventHtStd, "
        SQLQueryCols += "FarRadius,CloseRadius, uuid,"
        SQLQueryCols += "EventLat2, EventLat2Std, EventLon2, EventLon2Std,EventHt2, EventHt2Std, "
        SQLQueryCols += "EventAzim, EventAzimStd, EventElev, EventElevStd, EventElevIsMax, RespondTo"
        SQLStatement += SQLQueryCols
        SQLStatement += " \n"
        SQLStatement += "FROM event_monitor "
        SQLStatement += "WHERE processedstatus = 0"

        cursor = self.db_conn.cursor().execute(SQLStatement)
        events = []

        # iterate through the rows, one row to an event

        for row in cursor:
            event = EventContainer(0, 0, 0, 0)
            col_count = 0
            colslist = SQLQueryCols.split(',')
            # iterate through the columns, one column to a value
            for col in row:
                event.setvalue(colslist[col_count].strip(), col)
                col_count += 1
                # this is the end of an event
            events.append(copy.copy(event))

        # iterated through all events
        return events

    def getfile(self, file_name, directory):

        """ Get the path to the file in the directory if it exists.
            If not, then return the path to ~/source/RMS


            Arguments:
                file_name: [string] name of file
                directory: [string] path to preferred directory

            Return:
                 file: [string] Path to platepar
        """

        file_list = []
        if os.path.isfile(os.path.join(directory, file_name)):
            file_list.append(str(os.path.join(directory, file_name)))
            return file_list
        else:

            if os.path.isfile(os.path.join(os.path.expanduser("~/source/RMS"), file_name)):
                file_list.append(str(os.path.join(os.path.expanduser("~/source/RMS"), file_name)))
                return file_list
        return []

    def getplateparfilepath(self, event):

        """ Get the path to the best platepar from the directory matching the event time


            Arguments:
                event: [event]

            Return:
                file: [string] Path to platepar
        """

        file_list = []

        if len(self.getdirectorylist(event)) > 0:
            file_list += self.getfile("platepar_cmn2010.cal", self.getdirectorylist(event)[0])
        if len(file_list) != 0:
            return file_list[0]
        else:
            return False

    def getdirectorylist(self, event):

        """ Get the paths of directories which may contain files associated with an event

             Arguments:
                 event: [event]

             Return:
                 directorylist: [list of paths] List of directories
        """

        directorylist = []
        eventtime = convertgmntimetoposix(event.dt)

        # iterate across the folders in CapturedFiles and convert the directory time to posix time
        if os.path.exists(os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir)):
            for nightdirectory in os.listdir(
                    os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir)):
                directoryposixtime = convertgmntimetoposix(nightdirectory[7:22])

                # if the posix time representation is before the event, and within 16 hours add to the list of directories
                # most unlikely that a single event could be split across two directories, unless there was large time uncertainty
                if directoryposixtime < eventtime and (eventtime - directoryposixtime).total_seconds() < 16 * 3600:
                    directorylist.append(
                        os.path.join(os.path.expanduser(self.config.data_dir), self.config.captured_dir,
                                     nightdirectory))
        return directorylist

    def findeventfiles(self, event, directorylist, fileextensionlist):

        """Take an event, directory list and an extension list and return paths to files

           Arguments:
                event: [event] Event of interest
                directorylist: [list of paths] List of directories which may contain the files sought
                fileextensionlist: [list of extensions] List of file extensions such as .fits, .bin

           Return:
                file_list: [list of paths] List of paths to files
        """
        try:
            eventtime = parser.parse(event.dt)
        except:
            eventtime = convertgmntimetoposix(event.dt)

        file_list = []
        # Iterate through the directory list, appending files with the correct extension
        for directory in directorylist:
            for fileextension in fileextensionlist:
                for file in os.listdir(directory):
                    if file.endswith(fileextension):
                        fileposixtime = convertgmntimetoposix(file[10:25])
                        if abs((fileposixtime - eventtime).total_seconds()) < event.timetolerance:
                            file_list.append(os.path.join(directory, file))

        return file_list

    def getfilelist(self, event):

        """Take an event, return paths to files

           Arguments:
               event: [event] Event of interest


           Return:
               file_list: [list of paths] List of paths to files
        """

        file_list = []

        file_list += self.findeventfiles(event, self.getdirectorylist(event), [".fits", ".bin"])
        if len(self.getdirectorylist(event)) > 0:
            file_list += self.getfile(".config", self.getdirectorylist(event)[0])
            file_list += self.getfile("platepar_cmn2010.cal", self.getdirectorylist(event)[0])

        return file_list

    def calculateclosestpoint(self, beg_lat, beg_lon, beg_ele, end_lat, end_lon, end_ele, ref_lat, ref_lon, ref_ele):

        """
        Calculate the closest approach of a trajectory to a reference point

        refer to https://globalmeteornetwork.groups.io/g/main/topic/96374390#8250


        Args:
            beg_lat: [float] Starting latitude of the trajectory
            beg_lon: [float] Starting longitude of the trajectory
            beg_ele: [float] Beginning height of the trajectory
            end_lat: [float] Ending latitude of the trajectory
            end_lon: [float] Ending longitude of the trajectory
            end_ele: [float] Ending height of the trajectory
            ref_lat: [float] Station latitude
            ref_lon: [float] Station longitude
            ref_ele: [float] Station height

        Returns:
            start_dis: Distance from station to start of trajectory
            end_dist: Distance from station to end of trajectory
            closest_dist: Distance at the closest point (possibly outside the start and end)

        """

        # Convert coordinates to ECEF
        beg_ecef = np.array(latLonAlt2ECEF(np.radians(beg_lat), np.radians(beg_lon), beg_ele))
        end_ecef = np.array(latLonAlt2ECEF(np.radians(end_lat), np.radians(end_lon), end_ele))
        ref_ecef = np.array(latLonAlt2ECEF(np.radians(ref_lat), np.radians(ref_lon), ref_ele))

        traj_vec = vectNorm(end_ecef - beg_ecef)
        start_vec, end_vec = (ref_ecef - beg_ecef), (ref_ecef - end_ecef)
        start_dist, end_dist = (np.sqrt((np.sum(start_vec ** 2)))), (np.sqrt((np.sum(end_vec ** 2))))

        # Consider whether vector is zero length by looking at start and end
        if [beg_lat, beg_lon, beg_ele] != [end_lat, end_lon, end_ele]:
         # Vector start and end points are different, so possible to
         # calculate the projection of the reference vector onto the trajectory vector
         proj_vec = beg_ecef + np.dot(start_vec, traj_vec) * traj_vec

         # Hence, calculate the vector at the nearest point, and the closest distance
         closest_vec = ref_ecef - proj_vec
         closest_dist = (np.sqrt(np.sum(closest_vec ** 2)))

        else:

         # Vector has zero length, do not try to calculate projection
         closest_dist = start_dist

        return start_dist, end_dist, closest_dist

    def trajectoryvisible(self, rp, event):

        """
        Given a platepar and an event, calculate the centiles of the trajectory which would be in the FoV.
        Working is in ECI, relative to the station coordinates.

        Args:
            rp: [platepar] reference platepar
            event: [event] event of interest

        Returns:
            pointsinfov: [integer] the number of points out of 100 in the field of view
            startdistance: [float] the distance in metres from the station to the trajectory start
            startangle: [float] the angle between the vector from the station to start of the trajectory
                        and the vector of the centre of the FOV
            enddistance: [float] the distance in metres from the station to the trajectort end
            endangle: [float] the angle between the vector from the station to end of the trajectory
                        and the vector of the centre of the FOV
            fovra: [float]  field of view Ra (degrees)
            fovdec: [float] fovdec of view Dec (degrees)

        """
        # Calculate diagonal FoV of camera
        diagonal_fov = np.sqrt(rp.fov_v ** 2 + rp.fov_h ** 2)

        # Calculation origin will be the ECI of the station taken from the platepar
        juldate = datetime2JD(convertgmntimetoposix(event.dt))
        origin = np.array(geo2Cartesian(rp.lat, rp.lon, rp.elev, juldate))

        # Convert trajectory coordinates to cartesian ECI at JD of event
        traj_stapt = np.array(geo2Cartesian(event.lat, event.lon, event.ht * 1000, juldate))
        traj_endpt = np.array(geo2Cartesian(event.lat2, event.lon2, event.ht2 * 1000, juldate))

        # Make relative to station coordinates
        stapt_rel, endpt_rel = traj_stapt - origin, traj_endpt - origin

        # trajectory vector, and vector for traverse
        traj_vec = traj_endpt - traj_stapt
        traj_inc = traj_vec / 100

        # the az_centre, alt_centre of the camera
        az_centre, alt_centre = platepar2AltAz(rp)

        # calculate Field of View RA and Dec at event time, and
        fovra, fovdec = altAz2RADec(az_centre, alt_centre, juldate, rp.lat, rp.lon)

        fovvec = np.array(raDec2ECI(fovra, fovdec))

        # iterate along the trajectory counting points in the field of view
        pointsinfov = 0
        for i in range(0, 100):
            point = (stapt_rel + i * traj_inc)
            point_fov = np.degrees(angularSeparationVect(vectNorm(point), vectNorm(fovvec)))
            if point_fov < diagonal_fov / 2:
                pointsinfov += 1

        # calculate some additional information for confidence
        startdistance = (np.sqrt(np.sum(stapt_rel ** 2)))
        startangle = math.degrees(angularSeparationVect(vectNorm(stapt_rel), vectNorm(fovvec)))
        enddistance = (np.sqrt(np.sum(endpt_rel ** 2)))
        endangle = math.degrees(angularSeparationVect(vectNorm(endpt_rel), vectNorm(fovvec)))

        return pointsinfov, startdistance, startangle, enddistance, endangle, fovra, fovdec

    def trajectorythroughfov(self, event):

        """
        For the trajectory contained in the event, calculate if it passed through the FoV defined by the platepar
        of the time of the event

        Args:
            event: [event] Calculate if the trajectory of this event passed through the field of view

        Returns:
            pointsinfov: [integer] Number of points of the trajectory split into 100 parts
                                   apparently in the FOV of the camera
            start_distance: [float] Distance from station to the start of the trajectory
            start_angle: [float] Angle from the centre of the FoV to the start of the trajectory
            end_distance: [float] Distance from station to the end of the trajectory
            end_angle: [float] Angle from the centre of the FoV to the end of the trajectory
        """

        # Read in the platepar for the event
        rp = Platepar()
        if not rp.read(self.getplateparfilepath(event)):
            rp.read(os.path.abspath('.'))

        pointsinfov, startdistance, startangle, enddistance, endangle, fovra, fovdec = self.trajectoryvisible(rp, event)
        return pointsinfov, startdistance, startangle, enddistance, endangle, fovra, fovdec

    def doupload(self, event, evcon, filelist, keepfiles = False, noupload = False, testmode = False):

        """Move all the files to a single directory. Make MP4s, stacks and jpgs
           Archive into a bz2 file and upload, using rsync. Delete all working folders.

        Args:
            event: [event] the event to be uploaded
            evcon: [path] path to the config file for the event
            filelist: [list of paths] the files to be uploaded
            keepfiles: [bool] keep the files after upload
            noupload: [bool] if True do everything apart from uploading
            testmode: [bool] if True prevents upload

        Returns:

        """

        eventmonitordirectory = os.path.expanduser(os.path.join(self.syscon.data_dir, "EventMonitor"))
        uploadfilename = "{}_{}".format(evcon.stationID, event.dt)
        thiseventdirectory = os.path.join(eventmonitordirectory, uploadfilename)

        # get rid of the eventdirectory, should never be needed
        if not keepfiles:
            if os.path.exists(thiseventdirectory) and eventmonitordirectory != "" and uploadfilename != "":
                shutil.rmtree(thiseventdirectory)

        # create a new event directory
        if not os.path.exists(thiseventdirectory):
            os.makedirs(thiseventdirectory)

        # put all the files from the filelist into the event directory
        for file in filelist:
            shutil.copy(file, thiseventdirectory)

        # make a stack
        stackFFs(thiseventdirectory, "jpg", captured_stack=True)

        if True:
            batchFFtoImage(os.path.join(eventmonitordirectory, uploadfilename), "jpg", add_timestamp=True,
                           ff_component='maxpixel')

        with open(os.path.join(eventmonitordirectory, uploadfilename, "event_report.txt"), "w") as info:
            info.write(event.eventtostring())

        # remove any leftover .bz2 files
        if not keepfiles:
            if os.path.isfile(os.path.join(eventmonitordirectory, "{}.tar.bz2".format(uploadfilename))):
                os.remove(os.path.join(eventmonitordirectory, "{}.tar.bz2".format(uploadfilename)))

        if not testmode:
            if os.path.isdir(eventmonitordirectory) and uploadfilename != "":
             log.info("Making archive of {}".format(os.path.join(eventmonitordirectory, uploadfilename)))
             log.info("Base name : {}".format(uploadfilename))
             log.info("Root dir  : {}".format(eventmonitordirectory))
             log.info("Base dir  : {}".format(os.path.join(eventmonitordirectory,uploadfilename)))
             archive_name = shutil.make_archive(uploadfilename, 'bztar', eventmonitordirectory, os.path.join(eventmonitordirectory,uploadfilename))
            else:
             log.info("Not making an archive of {}, not sensible.".format(os.path.join(eventmonitordirectory, uploadfilename)))

        # Remove the directory where the files were assembled
        if not keepfiles:
            if os.path.exists(thiseventdirectory) and thiseventdirectory != "":
                shutil.rmtree(thiseventdirectory)

        if not noupload and not testmode:
         archives = glob(os.path.join(eventmonitordirectory,"*.bz2"))
         upload_status = uploadSFTP(self.syscon.hostname, self.syscon.stationID.lower(),eventmonitordirectory,self.syscon.event_monitor_remote_dir,archives,rsa_private_key=self.config.rsa_private_key)
         # set to the fast check rate after an upload
         self.check_interval = self.syscon.event_monitor_check_interval_fast
         log.info("Now checking at {:2.2f} minute intervals".format(self.check_interval))
         pass
        else:
         upload_status = False


        # Remove the directory
        if not keepfiles and upload_status:
            shutil.rmtree(eventmonitordirectory)

    def checkevents(self, evcon,  testmode = False):

        """
        Args:
            evcon: configuration object at the time of this event

        Returns:
            Nothing
        """

        # Get the work to be done
        unprocessed = self.getUnprocessedEventsfromDB()

        # Iterate through the work
        for event in unprocessed:

            # Get the files
            file_list = self.getfilelist(event)

            # If there are no files, then mark as processed and continue
            if (len(file_list) == 0 or file_list == [None]) and not testmode:
                log.info("No files for event at {}".format(event.dt))
                self.markeventasprocessed(event)
                continue

            # If there is a .config file then parse it as evcon - not the station config
            for file in file_list:
                if file.endswith(".config"):
                    evcon = cr.parse(file)

            # From the infinitely extended trajectory, work out the closest point to the camera
            start_dist, end_dist, atmos_dist = self.calculateclosestpoint(event.lat, event.lon, event.ht * 1000,
                                                                          event.lat2,
                                                                          event.lon2, event.ht2 * 1000, evcon.latitude,
                                                                          evcon.longitude, evcon.elevation)
            min_dist = min([start_dist, end_dist, atmos_dist])

            # If trajectory outside the farradius, do nothing, and mark as processed
            if min_dist > event.farradius * 1000 and not testmode:
                log.info("Event at {} was {:4.1f}km away, outside {:4.1f}km, so was ignored".format(event.dt,min_dist/1000, event.farradius))
                self.markeventasprocessed(event)
                # Do no more work
                continue


            # If trajectory inside the closeradius, then do the upload and mark as processed
            if min_dist < event.closeradius * 1000 and not testmode:
                # this is just for info
                log.info("Event at {} was {:4.1f}km away, inside {:4.1f}km so is uploaded with no further checks.".format(event.dt, min_dist / 1000, event.closeradius))
                count, event.startdistance, event.startangle, event.enddistance, event.endangle, event.fovra, event.fovdec = self.trajectorythroughfov(
                    event)
                self.doupload(event, evcon, file_list, testmode)
                self.markeventasprocessed(event)
                if len(file_list) > 0:
                    self.markeventasuploaded(event, file_list)
                # Do no more work
                continue


            # If trajectory inside the farradius, then check if the trajectory went through the FoV
            # The returned count is the number of 100th parts of the trajectory observed through the FoV
            if min_dist < event.farradius * 1000 or testmode:
                log.info("Event at {} was {:4.1f}km away, inside {:4.1f}km, consider FOV.".format(event.dt, min_dist / 1000, event.farradius))
                count, event.start_distance, event.start_angle, event.end_distance, event.end_angle, event.fovra, event.fovdec = self.trajectorythroughfov(event)
                if count != 0:
                    log.info("Event at {} had {} points out of 100 in the trajectory in the FOV. Uploading.".format(event.dt, count))
                    self.doupload(event, evcon, file_list, testmode=testmode)
                    self.markeventasuploaded(event, file_list)
                    if testmode:
                        rp = Platepar()
                        rp.read(self.getplateparfilepath(event))
                        with open(os.path.expanduser(os.path.join(self.syscon.data_dir, "testlog")), 'at') as logfile:
                            logfile.write(
                                "{} LOC {} Az:{:3.1f} El:{:3.1f} sta_lat:{:3.4f} sta_lon:{:3.4f} sta_dist:{:3.0f} end_dist:{:3.0f} fov_h:{:3.1f} fov_v:{:3.1f} sa:{:3.1f} ea::{:3.1f} \n".format(
                                    convertgmntimetoposix(event.dt), evcon.stationID, rp.az_centre, rp.alt_centre,
                                    rp.lat, rp.lon, event.start_distance / 1000, event.end_distance / 1000, rp.fov_h,
                                    rp.fov_v, event.start_angle, event.end_angle))
                else:
                    log.info("Event at {} did not pass through FOV.".format(event.dt))
                if not testmode:
                    self.markeventasprocessed(event)
                # Do no more work
                continue
        return None

    def start(self):
        """ Starts the event monitor. """

        super(EventMonitor, self).start()
        log.info("EventMonitor was started")

    def stop(self):
        """ Stops the event monitor. """

        self.db_conn.close()
        time.sleep(2)
        self.exit.set()
        self.join()
        log.info("EventMonitor has stopped")

    def geteventsandcheck(self, testmode=False):
        """
        Gets event(s) from the webpage, or a local file.
        Calls self.addevent to add them to the database
        Calls self.checkevents to see if the database holds any unprocessed events

        Args:
            testmode: [bool] if set true looks for a local file, rather than a web address

        Returns:
            Nothing
        """

        events = self.getEventsfromWebPage(testmode)
        for event in events:
            if event.checkreasonable():
                self.addevent(event)

        # Go through all events and check if they need to be uploaded - this iterates through the database
        self.checkevents(self.config, testmode=testmode)

    def run(self):


        # Delay to get everything else done first
        time.sleep(20)
        while not self.exit.is_set():

            log.info("EventMonitor webpage check starting")
            self.geteventsandcheck()
            log.info("EventMonitor webpage check completed")
            # Wait for the next check
            self.exit.wait(60 * self.check_interval)
            #We are running fast, but have not made an upload, then check more slowly next time
            if self.check_interval < self.syscon.event_monitor_check_interval:
                self.check_interval = self.check_interval * 1.1
                log.info("Check interval now set to {:2.2f} minutes".format(self.check_interval))



def convertgmntimetoposix(timestring):
    """
    Converts the filenaming time convention used by GMN into posix

    Args:
        timestring: [string] time represented as a string e.g. 20230527_032115

    Returns:
        posix compatible time
    """

    dt_object = datetime.strptime(timestring.strip(), "%Y%m%d_%H%M%S")
    return dt_object

#https://stackoverflow.com/questions/2030053/how-to-generate-random-strings-in-python
def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

def platepar2AltAz(rp):

    """

    Args:
        rp: Platepar

    Returns:
        Ra_d : [degrees] Ra of the platepar at its creation date
        dec_d : [degrees] Dec of the platepar at its creation date
        JD : [float] JD of the platepar creation
        lat : [float] lat of the station
        lon : [float] lon of the station

    """

    RA_d = np.radians(rp.RA_d)
    dec_d = np.radians(rp.dec_d)
    JD = rp.JD
    lat = np.radians(rp.lat)
    lon = np.radians(rp.lon)

    return np.degrees(cyTrueRaDec2ApparentAltAz(RA_d, dec_d, JD, lat, lon))


def raDec2ECI(ra, dec):

    """

    Convert right ascension and declination to Earth-centered inertial vector.

    Arguments:
        ra: [float] right ascension in degrees
        dec: [float] declination in degrees

    Return:
        (x, y, z): [tuple of floats] Earth-centered inertial coordinates in metres

    """

    x = np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z = np.sin(np.radians(dec))

    return x, y, z


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="""Check a web page for trajectories, and upload relevant data. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    arg_parser.add_argument('-o', '--oneshot', dest='one_shot', default=False, action="store_true",
                            help="Run once, and terminate.")

    arg_parser.add_argument('-d', '--deletedb', dest='delete_db', default=False, action="store_true",
                            help="Delete the event_monitor database at initialisation.")

    arg_parser.add_argument('-k', '--keepfiles', dest='keepfiles', default=False, action="store_true",
                            help="Keep working files")

    arg_parser.add_argument('-n', '--noupload', dest='noupload', default=False, action="store_true",
                            help="Do not upload")


    cml_args = arg_parser.parse_args()

    # Load the config file
    syscon = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('.'))

    # Set the web page to monitor


    try:
        # Add a random string after the URL to defeat caching
        web_page = urllib.request.urlopen(syscon.event_monitor_webpage + "?" + randomword(6)).read().decode("utf-8").splitlines()
    except:

        log.info("Nothing found at {}".format(syscon.event_monitor_webpage))


    if cml_args.delete_db and os.path.isfile(os.path.expanduser("~/RMS_data/event_monitor.db")):
        os.unlink(os.path.expanduser("~/RMS_data/event_monitor.db"))

    em = EventMonitor(syscon)



    if cml_args.one_shot:
        print("EventMonitor running once")
        em.geteventsandcheck()

    else:
        print("EventMonitor running indefinitely")
        em.start()

