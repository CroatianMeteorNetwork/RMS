""" Automatically uploads data files of an event as given on the website. """

from __future__ import print_function, division, absolute_import

import os
import sqlite3


import multiprocessing

import RMS.ConfigReader as cr


class EventContainer(object):
    def __init(self, dt, lat, lon, ht, 
               lat_sigma=0.2, lon_sigma=0.2, ht_sigma=10, 
               lat2=None, lon2=None, ht2=None,
               lat2_sigma=None, lon2_sigma=None, ht2_sigma=None,
               all_radius=150, fov_only_radius=600, time_tolerance=30):
        
        self.dt = dt
        self.lat = lat
        self.lon = lon
        self.ht = ht
        self.lat_sigma = lat_sigma
        self.lon_sigma = lon_sigma
        self.ht_sigma = ht_sigma
        
        # Optional second point
        self.lat2 = lat2
        self.lon2 = lon2
        self.ht2 = ht2
        self.lat2_sigma = lat2_sigma
        self.lon2_sigma = lon2_sigma
        self.ht2_sigma = ht2_sigma
        
        # Search parameters
        self.all_radius = all_radius
        self.fov_only_radius = fov_only_radius
        self.time_tolerance = time_tolerance

        # A list of files that were uploaded to the server
        self.files_uploaded = []

        # Datetime of when the event was checked
        self.time_completed = None

        # Whether the event was observed or not
        self.observed_status = None

        # Whether the event was processed or not
        self.processed_status = False



class EventMonitor(multiprocessing.Process):
    def __init__(self, config):
        """ Automatically uploads data files of an event (e.g. fireball) as given on the website.

        Arguments:
            config: [Config] Configuration object.
        """

        super(EventMonitor, self).__init__()
        self.config = config

        # The path to the event monitor data base
        self.event_monitor_db_path = os.path.join(os.path.abspath(self.config.data_dir), 
                                                  self.config.event_monitor_db_name)
        
        # Load the event monitor data base
        self.db_conn = self.loadEventMonitorDB()


    def loadEventMonitorDB(self):
        """ Loads the event monitor data base. """
        
        # Create the event monitor data base if it does not exist
        if not os.path.isfile(self.event_monitor_db_path):
            self.createEventMonitorDB()
        
        # Load the event monitor data base
        conn = sqlite3.connect(self.event_monitor_db_path)

        return conn
    

    def createEventMonitorDB(self):
        """ Creates the event monitor data base. """

        # Create the event monitor data base
        conn = sqlite3.connect(self.event_monitor_db_path)

        # Create the table
        conn.execute("""CREATE TABLE event_monitor (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            dt TEXT NOT NULL,
                            lat REAL NOT NULL,
                            lon REAL NOT NULL,
                            ht REAL NOT NULL,
                            lat_sigma REAL NOT NULL,
                            lon_sigma REAL NOT NULL,
                            ht_sigma REAL NOT NULL,
                            lat2 REAL,
                            lon2 REAL,
                            ht2 REAL,
                            lat2_sigma REAL,
                            lon2_sigma REAL,
                            ht2_sigma REAL,
                            all_radius REAL NOT NULL,
                            fov_only_radius REAL NOT NULL,
                            time_tolerance REAL NOT NULL,
                            files_uploaded TEXT,
                            time_completed TEXT,
                            observed_status BOOL,
                            processed_status BOOL
                            )""")

        # Commit the changes
        conn.commit()

        # Close the connection
        conn.close()


    def run(self):

        while not self.exit.is_set():

            # Check the event monitor web page for new events
            events = self.checkEventMonitorWebPage()

            # Go through all events and check if they need to be uploaded
            for event in events:
                if self.checkEvent(event):
                    self.uploadEvent(event)

                # Set the event as processed in the database
                self.setAsProcessed(event)

            # Wait for 30 minutes until the next check
            self.exit.wait(60*self.config.event_monitor_check_interval)



    def checkEventMonitorWebPage(self):
        """ Checks the event monitor web page for new events. """

        # Currently does nothing, but it should return a list of EventContainer objects that need to be
        # checked if they were observed
        
        # After parsing the file on the website, check the database to see if the event was already
        # checked. If not, add it to the database and only update its processed status once it was
        # checked.

        return None
    

    def checkEvent(self, event):
        """ Check if the given event needs to be uploaded. """

        # Check if the event should be uploaded

        return None
    

    def uploadEvent(self, event):
        """ Uploads the given  to the server. """

        # Upload the event

        return None
    

    def setAsProcessed(self, event):
        """ Sets the event as processed in the database. """

        # Set the event as processed in the database
        self.db_conn.execute("""UPDATE event_monitor SET processed_status = ? WHERE id = ?""",
                                (True, event.id))

        return None


    def start(self):
        """ Starts the event monitor. """

        super(EventMonitor, self).start()


    def stop(self):
        """ Stops the event monitor. """

        self.exit.set()
        self.join()




if __name__ == "__main__":
    
    # Load the default config file
    config = cr.Config()

    # Set the web page to monitor
    config.event_monitor_web_page = "globalmeteornetwork.org/data/event_watchlist.txt"

    # Name of the event monitor data base (sqlite3)
    config.event_monitor_db_name = "event_monitor.db"

    # Minutes to wait between website checks
    config.event_monitor_check_interval = 30

    # Name of the directory on the server where the event files will be uploaded
    config.event_monitor_upload_dir = "event_monitor"

