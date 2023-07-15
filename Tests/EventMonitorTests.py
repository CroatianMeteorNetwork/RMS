""""" Automatically uploads data files based on time and trajectory information given on a website. """

from __future__ import print_function, division, absolute_import

from RMS.EventMonitor import EventContainer, EventMonitor, convertGMNTimeToPOSIX
import RMS.ConfigReader as ConfigReader

import os
import time
import datetime
import argparse
import pathlib
from RMS.Formats.Platepar import Platepar
from datetime import datetime
from dateutil import parser
from tqdm import tqdm
from RMS.Astrometry.Conversions import AEH2Range

path_to_test_data = ""
platepars_test_data = "wget http://58.84.202.15:8243/data/platepars.tar.bz2 -O platepars.tar.bz2"
path_to_platepars = os.path.expanduser("~/RMS_data/test")
file_name = "platepars.tar.bz2"

def createATestEvent02():

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230522_183958")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", -33.681098)
    test_event.setValue("EventLatStd", 0.31)
    test_event.setValue("EventLon", 116.346892)
    test_event.setValue("EventLonStd", 0.32)
    test_event.setValue("EventHt", 109.6105)
    test_event.setValue("EventHtStd", 15)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", -33.790603)
    test_event.setValue("EventLat2Std", 0.33)
    test_event.setValue("EventLon2", 115.533457)
    test_event.setValue("EventLon2Std", 0.34)
    test_event.setValue("EventHt2", 72.9885)
    test_event.setValue("EventHt2Std", 0.35)

    test_uuid = "27e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event

def createATestEvent03():

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230525_190849")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", -33.558046)
    test_event.setValue("EventLatStd", 0.31)
    test_event.setValue("EventLon", 114.99615)
    test_event.setValue("EventLonStd", 0.32)
    test_event.setValue("EventHt", 92.2191)
    test_event.setValue("EventHtStd", 15)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", -33.564671)
    test_event.setValue("EventLat2Std", 0.33)
    test_event.setValue("EventLon2", 115.039056)
    test_event.setValue("EventLon2Std", 0.34)
    test_event.setValue("EventHt2", 79.9325)
    test_event.setValue("EventHt2Std", 0.35)

    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event

def createATestEvent04():

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230525_190849")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", -33.558046)
    test_event.setValue("EventLatStd", 0.31)
    test_event.setValue("EventLon", 114.99615)
    test_event.setValue("EventLonStd", 0.32)
    test_event.setValue("EventHt", 92.2191)
    test_event.setValue("EventHtStd", 15)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", -33.564671)
    test_event.setValue("EventLat2Std", 0.33)
    test_event.setValue("EventLon2", 115.039056)
    test_event.setValue("EventLon2Std", 0.34)
    test_event.setValue("EventHt2", 79.9325)
    test_event.setValue("EventHt2Std", 0.35)

    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event


def createATestEvent05():

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230517_111327")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", -33.478619)
    test_event.setValue("EventLatStd", 0.31)
    test_event.setValue("EventLon", 115.785164)
    test_event.setValue("EventLonStd", 0.32)
    test_event.setValue("EventHt", 97.5045)
    test_event.setValue("EventHtStd", 15)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", -32.694214)
    test_event.setValue("EventLat2Std", 0.33)
    test_event.setValue("EventLon2", 115.708743)
    test_event.setValue("EventLon2Std", 0.34)
    test_event.setValue("EventHt2", 81.0418)
    test_event.setValue("EventHt2Std", 0.35)

    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event

def createATestEvent06():

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230526_115827")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", -31.797397)
    test_event.setValue("EventLatStd", 0.31)
    test_event.setValue("EventLon", 118.132762)
    test_event.setValue("EventLonStd", 0.32)
    test_event.setValue("EventHt", 94.2331)
    test_event.setValue("EventHtStd", 15)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", -32.657057)
    test_event.setValue("EventLat2Std", 0.33)
    test_event.setValue("EventLon2", 118.349618)
    test_event.setValue("EventLon2Std", 0.34)
    test_event.setValue("EventHt2", 69.2098)
    test_event.setValue("EventHt2Std", 0.35)

    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event

def createATestEvent07():

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230526_205441")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", -32.263726)
    test_event.setValue("EventLatStd", 0.31)
    test_event.setValue("EventLon", 116.016066)
    test_event.setValue("EventLonStd", 0.32)
    test_event.setValue("EventHt", 89.0537)
    test_event.setValue("EventHtStd", 15)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", -32.187818)
    test_event.setValue("EventLat2Std", 0.33)
    test_event.setValue("EventLon2", 116.111370)
    test_event.setValue("EventLon2Std", 0.34)
    test_event.setValue("EventHt2", 80.8778)
    test_event.setValue("EventHt2Std", 0.35)

    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event

def createATestEvent08():

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230601_124235")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", 45)
    test_event.setValue("EventLatStd", 0)
    test_event.setValue("EventLon", 180)
    test_event.setValue("EventLonStd", 0)
    test_event.setValue("EventHt", 100)
    test_event.setValue("EventHtStd", 0)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", 0)
    test_event.setValue("EventLat2Std", 0)
    test_event.setValue("EventLon2", 0)
    test_event.setValue("EventLon2Std", 0)
    test_event.setValue("EventHt2", 0)
    test_event.setValue("EventHt2Std", 0)

    test_event.setValue("EventAzim", 0)
    test_event.setValue("EventElev", 0)



    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event

def createATestEvent09():

    test_event = EventContainer("", 0, 0, 0)
    test_event.setValue("EventTime", "20230710_134048")
    test_event.setValue("TimeTolerance", 60)
    test_event.setValue("EventLat", -31.247944)
    test_event.setValue("EventLatStd", 0)
    test_event.setValue("EventLon", 116.428754)
    test_event.setValue("EventLonStd", 0)
    test_event.setValue("EventHt", 86.7735)
    test_event.setValue("EventHtStd", 0)
    test_event.setValue("CloseRadius", 152)
    test_event.setValue("FarRadius", 153)

    test_event.setValue("EventLat2", 0)
    test_event.setValue("EventLat2Std", 0)
    test_event.setValue("EventLon2", 0)
    test_event.setValue("EventLon2Std", 0)
    test_event.setValue("EventHt2", 0)
    test_event.setValue("EventHt2Std", 0)

    test_event.setValue("EventAzim", 265.9)
    test_event.setValue("EventElev", 49.59)



    test_uuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    test_event.setValue("uuid", test_uuid)

    return test_event



def testEventContainer():

    testevent01 = EventContainer("20230522_182218", -33.692091, 115.378660, 102)
    if testevent01.dt != "20230522_182218" or testevent01.lat != -33.692091 or testevent01.lon != 115.378660 or testevent01.ht != 102:
        print("Event container initialisation testing failed")
        quit()



    if testevent01.dt != "20230522_182218" or testevent01.lat != -33.692091 or testevent01.lon != 115.378660 or testevent01.ht != 102:
        print("Event container scoping testing failed")
        quit()


    expectedresult =   ['# Required ',
                        'EventTime                : 20230522_183958',
                        'TimeTolerance (s)        : 60',
                        'EventLat (deg +N)        : -33.68',
                        'EventLatStd (deg)        : 0.31',
                        'EventLon (deg +E)        : 116.35',
                        'EventLonStd (deg)        : 0.32',
                        'EventHt (km)             : 109.61',
                        'EventHtStd (km)          : 15.00',
                        'CloseRadius(km)          : 152.00',
                        'FarRadius (km)           : 153.00',
                        '',
                        '# Optional second point      ',
                        'EventLat2 (deg +N)       : -33.79',
                        'EventLat2Std (deg)       : 0.33',
                        'EventLon2 (deg +E)       : 115.53',
                        'EventLon2Std (deg)       : 0.34',
                        'EventHt2 (km)            : 72.99',
                        'EventHtStd2 (km)         : 0.35',
                        '',
                        '# Or a trajectory instead    ',
                        'EventAzim (deg +E of N)  : 0.00',
                        'EventAzimStd (deg)       : 0.00',
                        'EventElev (deg)          : 0.00',
                        'EventElevStd (deg):      : 0.00',
                        'EventElevIsMax           : 0.00',
                        '',
                        '# Control information        ',
                        'StationsRequired         : ',
                        'uuid                     : 27e4a2d7-4111-4a72-8a30-969f71fc9207',
                        'RespondTo                : ',
                        '# Trajectory information     ',
                        'Start Distance (km)      : 0.00',
                        'Start Angle              : 0.00',
                        'End Distance (km)        : 0.00',
                        'End Angle                : 0.00',
                        '# Station information        ',
                        'Field of view RA         : 0.00',
                        'Field of view Dec        : 0.00',
                        '',

                        'END',
                        '' ]
    test_event_02 = createATestEvent02()

    if test_event_02.eventToString().split("\n") != expectedresult:
        print("Test event 02 event set values or eventToString has failed")
        print(expectedresult)
        print(test_event_02.eventToString().split("\n"))
        quit()






def testDBFunctions(em):

    operational_config = ConfigReader.loadConfigFromDirectory(cml_args.config, os.path.abspath(''))


    em.createEventMonitorDB(test_mode = True)
    #test calling twice
    em.createEventMonitorDB(test_mode = True)
    db_path = (os.path.expanduser( os.path.join(operational_config.data_dir, operational_config.event_monitor_db_name)))
    db_file = pathlib.Path(db_path)
    if os.path.exists(db_file):
      db_file.unlink()
    em.createEventMonitorDB(test_mode = True)
    em.addEvent(createATestEvent02())
    expected_result = ['# Required ',
                      'EventTime                : 20230522_183958',
                      'TimeTolerance (s)        : 60',
                      'EventLat (deg +N)        : -33.68',
                      'EventLatStd (deg)        : 0.31',
                      'EventLon (deg +E)        : 116.35',
                      'EventLonStd (deg)        : 0.32',
                      'EventHt (km)             : 109.61',
                      'EventHtStd (km)          : 15.00',
                      'CloseRadius(km)          : 152.00',
                      'FarRadius (km)           : 153.00',
                      '',
                      '# Optional second point      ',
                      'EventLat2 (deg +N)       : -33.79',
                      'EventLat2Std (deg)       : 0.33',
                      'EventLon2 (deg +E)       : 115.53',
                      'EventLon2Std (deg)       : 0.34',
                      'EventHt2 (km)            : 72.99',
                      'EventHtStd2 (km)         : 0.35',
                      '',
                      '# Or a trajectory instead    ',
                      'EventAzim (deg +E of N)  : 0.00',
                      'EventAzimStd (deg)       : 0.00',
                      'EventElev (deg)          : 0.00',
                      'EventElevStd (deg):      : 0.00',
                      'EventElevIsMax           : 0.00',
                      '',
                      '# Control information        ',
                      'StationsRequired         : ',
                      'uuid                     : 27e4a2d7-4111-4a72-8a30-969f71fc9207',
                      'RespondTo                : ',
                      '# Trajectory information       ',
                      'Start Distance (km)      : 0.00',
                      'Start Angle              : 0.00',
                      'End Distance (km)        : 0.00',
                      'End Angle                : 0.00',
                      '# Station information        ',
                      'Field of view RA         : 0.00',
                      'Field of view Dec        : 0.00',
                      '',
                      'END']

    event_list = em.getUnprocessedEventsfromDB()

    match = True
    line_count = 0
    for line in event_list[0].eventToString().split("\n"):
     if event_list[0].eventToString().split("\n")[line_count] != expected_result[line_count]:

        print(event_list[0].eventToString().split("\n")[line_count])
        print(expected_result[line_count])
        quit()

     if match:
         print("DB retrieval success")
     else:
         print("DB retrieval fail")
         quit()

    if em.addEvent(createATestEvent02()):
        print("Quit - added same event twice")
        quit()
    else:
        print("DB rejected duplicate event success")
    pass

    em.markEventAsProcessed(event_list[0])


def testClosestPoint():

    print("Path to platepars {}".format(path_to_platepars))
    shell_command = "mkdir -p {} ; \n".format(path_to_platepars)
    shell_command += "cd {} ; \n".format(path_to_platepars)
    shell_command += platepars_test_data + "  ; \n"
    shell_command += "tar -xf {}  ; ".format(os.path.join(path_to_platepars, file_name))
    shell_command += "rm {}       ; ".format(os.path.join(path_to_platepars, file_name))
    print(shell_command)
    os.system(shell_command)
    print(os.path.join(path_to_platepars, "platepars", "au0006", ".config"))
    op_con = ConfigReader.parse(os.path.join(path_to_platepars, "platepars", "au0006", ".config"), os.path.abspath('eventmonitor'))
    shell_command = "cd {} ; ".format(path_to_platepars)
    shell_command += "rm -rf platepars"
    em = EventMonitor(op_con)
    em.addEvent(createATestEvent03())
    events = em.getUnprocessedEventsfromDB()
    e=events[0]

    sd, ed, cd = em.calculateclosestpoint(e.lat,e.lon,e.ht *1000 ,e.lat2,e.lon2,e.ht2 *1000 ,op_con.latitude, op_con.longitude, op_con.elevation)

    # results calculated using http://cosinekitty.com/compass.html
    expected_sd = 179.976# km
    expected_ed = 172.718# km

    if abs(expected_sd - sd/1000) > 0.1:
        print("Start distance calculation failed")
        quit()
    else:
        print("Start distance calculation success")

    if abs(expected_ed - ed/1000) > 0.1:
        print("End distance calculation failed")
        quit()
    else:
        print("End distance calculation success")

def testTrajectoryThroughFOVQuick(em):

    print("Path to platepars {}".format(path_to_platepars))
    au0006_platepars = os.path.join(path_to_platepars, "platepars", "au0006")
    au0009_platepars = os.path.join(path_to_platepars, "platepars", "au0009")
    au000k_platepars = os.path.join(path_to_platepars, "platepars", "au000k")
    shell_command = "cd {} ; ".format(path_to_platepars)
    shell_command += "tar -xf {}  ; ".format(os.path.join(path_to_platepars, file_name))
    os.system(shell_command)
    test_rp1 = Platepar()
    print(os.path.join(au0006_platepars, "platepar_cmn2010.cal"))
    test_rp1.read(os.path.join(au0006_platepars, "platepar_cmn2010.cal"))
    op_con = ConfigReader.parse(os.path.join(au0006_platepars, ".config"), os.path.abspath('eventmonitor'))

    #Test a trajectory that is inside the FoV of AU0006.

    em.addEvent(createATestEvent03())

    pi_fov, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1, em.getUnprocessedEventsfromDB()[0])

    if pi_fov != 100:
        print("FOV calculations failed - AU0006 false negative")
        quit()
    else:
        print("FOV calculation success - AU0006 true positive")

    # AU0009 was colocated with AU0006 but slightly different pointing and did not see this event
    test_rp1.read(os.path.join(os.path.expanduser(au0009_platepars), "platepar_cmn2010.cal"))
    op_con = ConfigReader.parse(os.path.join(os.path.expanduser(au0009_platepars), ".config"), os.path.abspath('eventmonitor'))

    pi_fov, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                em.getUnprocessedEventsfromDB()[0])

    if pi_fov != 0:
        print("FOV calculations failed - AU0009 false positive")
        quit()
    else:
        print("FOV calculation success - AU0009 true negative")

    # Test a trajectory that should have been seen by AU0006 and AU0009
    # 20230517_111327
    # Refer to
    # https://globalmeteornetwork.org/weblog//AU/AU0006/AU0006_20230517_095155_165765_detected/AU0006_20230517_095155_165765_DETECTED_thumbs.jpg
    # and
    # https://globalmeteornetwork.org/weblog//AU/AU0009/AU0009_20230517_095215_180059_detected/AU0009_20230517_095215_180059_DETECTED_thumbs.jpg

    test_rp1.read(os.path.join(os.path.expanduser(au0006_platepars), "platepar_cmn2010.cal"))
    op_con = ConfigReader.parse(os.path.join(os.path.expanduser(au0006_platepars), ".config"), os.path.abspath(''))

    em.addEvent(createATestEvent05())
    pi_FOV_06, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                  em.getUnprocessedEventsfromDB()[1])

    test_rp1.read(
        os.path.join(os.path.expanduser(au0009_platepars), "platepar_cmn2010.cal"))
    op_con = ConfigReader.parse(os.path.join(os.path.expanduser(au0009_platepars), ".config"), os.path.abspath(''))

    em.addEvent(createATestEvent05())
    pi_FOV_09, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                  em.getUnprocessedEventsfromDB()[1])

    if pi_FOV_06 != 100 or pi_FOV_09 != 100:
        print("FOV calculations failed - AU0006 or AU0009 false negative")
        quit()
    else:
        print("FOV calculation success - AU0006 and AU0009 true positive")

    # now test a trajectory that is outside both cameras FOV

    # 20230526_115827

    # refer to
    # https://globalmeteornetwork.org/weblog//AU/AU001B/AU001B_20230526_094222_772451_detected/AU001B_20230526_094222_772451_DETECTED_thumbs.jpg

    em.addEvent(createATestEvent06())

    test_rp1.read(
        os.path.join(os.path.expanduser(au0009_platepars), "platepar_cmn2010.cal"))
    op_con = ConfigReader.parse(os.path.join(os.path.expanduser(au0009_platepars), ".config"), os.path.abspath(''))

    pi_FOV_09, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                  em.getUnprocessedEventsfromDB()[2])

    test_rp1.read(
        os.path.join(os.path.expanduser(au0006_platepars), "platepar_cmn2010.cal"))
    op_con = ConfigReader.parse(os.path.join(os.path.expanduser(au0006_platepars), ".config"), os.path.abspath(''))


    pi_FOV_06, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1, em.getUnprocessedEventsfromDB()[2])

    if pi_FOV_06 != 0 or pi_FOV_09 != 0:
        print("Fov Calculations failed - AU0006 or AU0009 false postive")
        quit()
    else:
        print("Fov Calculation success - AU0006 and AU0009 true negative")



    # try a non-standard camera AU000K
    # 20230526_205441
    # refer to https://globalmeteornetwork.org/weblog//AU/AU000K/AU000K_20230526_094747_162584_detected/AU000K_20230526_094747_162584_DETECTED_thumbs.jpg

    test_rp1.read(
        os.path.join(os.path.expanduser(au000k_platepars), "platepar_cmn2010.cal"))
    op_con = ConfigReader.parse(os.path.join(os.path.expanduser(au000k_platepars), ".config"), os.path.abspath(''))

    em.addEvent(createATestEvent07())
    pi_FOV_0K, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1, em.getUnprocessedEventsfromDB()[3])
    if pi_FOV_0K != 100:
        print("Fov Calculations failed - AU000K false negative")
        quit()
    else:
        print("Fov Calculation success - AU000K true positive")

    # now fake the camera alt
    test_rp1.alt_centre += 10
    pi_FOV_0K, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pi_FOV_0K == 100:
        print("Fov Calculations failed - AU000K with a fake offset returned same value")
        print("Safe to ignore this fail, because the Alt and Az are now calculated from the Ra Dec")
    else:
        print("Fov Calculation success - AU000K returned only partial trajectory - {}%".format(pi_FOV_0K))

    test_rp1.az_centre += 10
    pi_FOV_0K, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pi_FOV_0K == 100:
        print("Fov Calculations failed - AU000K with a fake offset returned same value")
        print("Safe to ignore this fail, because the Alt and Az are now calculated from the Ra Dec")
    else:
        print("Fov Calculation success - AU000K returned only partial trajectory - {}%".format(pi_FOV_0K))

    test_rp1.az_centre += 10
    pi_FOV_0K, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pi_FOV_0K !=0 :
        print("Fov Calculations failed - AU000K returned points when incorrect az")
        print("Safe to ignore this fail, because the Alt and Az are now calculated from the Ra Dec")
    else:
        print("Fov Calculation success - AU000K returned no trajectory visible")

    if pi_FOV_0K == 100:
        print("Fov Calculations failed - AU000K with a fake offset returned same value")
        print("Safe to ignore this fail, because the Alt and Az are now calculated from the Ra Dec")
    else:
        print("Fov Calculation success - AU000K returned only partial trajectory - {}%".format(pi_FOV_0K))

    test_rp1.RA_d  += 10
    pi_FOV_0K, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pi_FOV_0K == 100:
        print("Fov Calculations failed - AU000K with a fake offset returned same value")
        quit()
    else:
        print("Fov Calculation success - AU000K returned only partial trajectory - {}%".format(pi_FOV_0K))

    test_rp1.RA_d += 10
    pi_FOV_0K, sd, sa, ed, ea, FOV_ra, FOV_dec = em.trajectoryVisible(test_rp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pi_FOV_0K != 0:
        print("Fov Calculations failed - AU000K returned points when incorrect RA")
        quit()
    else:
        print("Fov Calculation success - AU000K returned no trajectory visible")

    shell_command = "cd {} ; ".format(path_to_platepars)
    shell_command += "rm -rf platepars"
    print(shell_command)
    os.system(shell_command)


def fullSystemTest(config, em):

    """
    Carries a check of the whole system against a snapshot of the GMN database, using a subset of camera information

    Args:
        config: The config for the local system - this is so that the correct file location can be used

    Returns:

    """


    if os.path.isfile(os.path.expanduser("~/RMS_data/test/testlog")):
         os.unlink(os.path.expanduser("~/RMS_data/test/testlog"))

    path_to_platepars = os.path.expanduser("~/RMS_data/test")
    file_name = "archives.tar.bz2"
    print("Path to platepars {}".format(path_to_platepars))
    shell_command = "cd {} ; \n".format(path_to_platepars)

    shell_command += "wget --quiet http://58.84.202.15:8243/data/archives.tar.bz2 -O archives.tar.bz2 \n"
    shell_command += "tar -xf {}  ; \n".format(os.path.join(path_to_platepars,file_name))
    print(shell_command)
    os.system(shell_command)

    file_name = "trajectorydata.tar.bz2"
    shell_command += "wget --quiet http://58.84.202.15:8243/data/trajectorydata.tar.bz2 -O trajectorydata.tar.bz2 \n"
    shell_command += "tar -xf {}  ; \n".format(os.path.join(path_to_platepars,file_name))
    print(shell_command)
    os.system(shell_command)


    camera_data_available = os.listdir(os.path.join(path_to_platepars,"archives"))
    camera_data_available.sort()
    trajectory_files = os.listdir(os.path.expanduser(os.path.join(path_to_platepars,"trajectorydata")))
    trajectory_files.sort()
    trajectory_list = []

    for trajectory_file in tqdm(trajectory_files):

        if trajectory_file[0:12] != "traj_summary":
            continue
        with open(os.path.join(os.path.join(path_to_platepars,"trajectorydata/"),trajectory_file)) as fh:

          filetowrite = False
          for trajectory in fh:
            if trajectory[0] != "#" :

              if trajectory.strip() != "" and trajectory.strip() != "\n":

                trajectory_list.append(trajectory.strip())

    if os.path.exists(os.path.expanduser('~/RMS_data/event_watchlist.txt')):
        os.unlink(os.path.expanduser('~/RMS_data/event_watchlist.txt'))

    if os.path.exists(os.path.expanduser(os.path.join(config.data_dir, "testlog"))):
        os.unlink(os.path.expanduser(os.path.join(config.data_dir, "testlog")))


    log_file = open(os.path.expanduser(os.path.join(config.data_dir, "testlog")), 'wt')
    for traj in tqdm(trajectory_list):
        relevant_trajectory = False
        #create a fresh event_watchlist.txt page
        #need to open as write so that we overwrite any previous trajectories


        t = traj.split(';')
        event_time, traj_cam   = t[2] , t[85].strip().split(",")
        lat_beg,lon_beg, ht_beg = t[63], t[65], t[67]
        lat_end,lon_end, ht_end = t[69], t[71], t[73]
        time.sleep(0.0001)

        # Check 1 - are observations that were seen predicted correctly
        # Iterate through all the cameras GMN associated with the event
        for camera in traj_cam:
                    # If any single camera in that event is in the local database
                    if camera.lower() in camera_data_available:
                        # And we have not already written an even_watchlist for this trajectory
                        if not relevant_trajectory:
                               #write all the cameras that saw the event to log file


                               with open(os.path.expanduser(os.path.join(em.syscon.data_dir, "testlog")), 'at') as log_file:
                                log_file.write("{} GMN {}  \n".format(event_time.strip().split('.')[0],traj_cam))

                               # and write an event_watch.txt file
                               # prepare an event and write to the local file for testing
                               tev = EventContainer(0,0,0,0)
                               tev.dt, tev.time_tolerance      = parser.parse(event_time).strftime("%Y%m%d_%H%M%S"), 5
                               tev.lat, tev.lon, tev.ht       = float(lat_beg), float(lon_beg), float(ht_beg)
                               tev.lat2, tev.lon2, tev.ht2    = float(lat_end), float(lon_end), float(ht_end)
                               tev.close_radius, tev.far_radius = 1, 1000

                               with open(os.path.expanduser('~/RMS_data/event_watchlist.txt'), 'wt') as watch_fh:
                                 watch_fh.write(tev.eventToString() + "\n")

                                 relevant_trajectory = True

        # For every camera, we need to check whether it saw the trajectory
        # This can't be a subsection of main loop, otherwise we only check for cameras that were used in the solver
        for camera in camera_data_available:

         if relevant_trajectory:

          ev_con = ConfigReader.Config()
          ev_con.data_dir = os.path.join(path_to_platepars,"archives",camera.lower(),"RMS_data")
          em.config.data_dir = os.path.join(path_to_platepars,"archives",camera.lower(),"RMS_data")
          #and write the camera name
          ev_con.stationID = camera.upper()
          # set the system config in eventmonitor
          em.syscon.data_dir = "~/RMS_data"
          em.getEventsAndCheck(testmode=True)

        unprocessed_events = em.getUnprocessedEventsfromDB()
        # because we are in testmode, the events do not get marked as processed,
        for unprocessed_event in unprocessed_events:
            em.markEventAsProcessed(unprocessed_event)

    shell_command = "cd {} ; ".format(path_to_platepars)
    os.system(shell_command)

def testAEH2Range():

    print("Testing AEH2Range")

    # Test a line straight up from ground level
    lat, lon, alt = 45, 45, 100000
    azim, elev, ht = 30, -5, 100000


    range_by_law_of_sines = AEH2Range(azim,elev,ht, lat, lon, alt, False)
    print("Range calculated by law of sines       {}".format(range_by_law_of_sines))
    range_by_optimised_solution = AEH2Range(azim, elev, ht, lat, lon, alt, True)
    print("Range calculated by optimised solution {}".format(range_by_optimised_solution))


def testIndividuals():



    if convertGMNTimeToPOSIX("20210925_163127") == datetime(2021, 9, 25, 16, 31, 27):
        print("convertgmntimetoposix success")
    else:
        print("convertgmntimetoposix fail")
        quit()

    event = createATestEvent08()
    #print(event.eventToString())
    event.transformToLatLon()
    print("Az:{} El:{}".format(event.azim, event.elev))
    #print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 90)
    event.setValue("EventElev", 0)
    event.transformToLatLon()
    print("Az:{} El:{}".format(event.azim, event.elev))
    #print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 180)
    event.setValue("EventElev", 0)
    event.transformToLatLon()
    print("Az:{} El:{}".format(event.azim, event.elev))
    #print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 270)
    event.setValue("EventElev", 0)
    event.transformToLatLon()
    print("Az:{} El:{}".format(event.azim, event.elev))
    #print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 0)
    event.setValue("EventElev", 45)
    event.transformToLatLon()
    print("Az:{} El:{}".format(event.azim, event.elev))
    #print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 0)
    event.setValue("EventElev", 90)
    event.transformToLatLon()
    print("Az:{} El:{}".format(event.azim, event.elev))
    #print(event.eventToString())
    event = createATestEvent08()
    event.setValue("EventAzim", 30)
    event.setValue("EventElev", 78)
    event.transformToLatLon()
    print("Az:{} El:{}".format(event.azim, event.elev))
    #print(event.eventToString())

    event = createATestEvent09()
    event.transformToLatLon()
    print("Az:{} El:{}".format(event.azim, event.elev))
    #print(event.eventToString())




def functionTest(config, em):

    #testAEH2Range()
    testIndividuals()
    #testEventContainer()
    #testDBFunctions(em)
    #testClosestPoint()
    #testTrajectoryThroughFOVQuick(em)
    #fullSystemTest(config, em)
    quit()

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="""Test procedure for Event Monitor. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    cml_args = arg_parser.parse_args()

    config = ConfigReader.loadConfigFromDirectory(cml_args.config, os.path.abspath('eventmonitor'))
    em = EventMonitor(config)
    em.syscon = ConfigReader.loadConfigFromDirectory(cml_args.config, os.path.abspath('eventmonitor'))


    functionTest(config, em)