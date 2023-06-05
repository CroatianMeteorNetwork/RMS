""""" Automatically uploads data files based on time and trajectory information given on a website. """

from __future__ import print_function, division, absolute_import

from RMS.EventMonitor import EventContainer, EventMonitor, convertgmntimetoposix
import RMS.ConfigReader as cr

import os
import time
import datetime
import argparse
import pathlib
from RMS.Formats.Platepar import Platepar
from datetime import datetime
from dateutil import parser
from tqdm import tqdm


pathtotestdata = ""
plateparstestdata = "wget http://58.84.202.15:8243/data/platepars.tar.bz2 -O platepars.tar.bz2"
pathtoplatepars = os.path.expanduser("~/RMS_data/test")
filename = "platepars.tar.bz2"

def createatestevent02():

    testevent = EventContainer("", 0, 0, 0)
    testevent.setvalue("EventTime", "20230522_183958")
    testevent.setvalue("TimeTolerance", 60)
    testevent.setvalue("EventLat", -33.681098)
    testevent.setvalue("EventLatStd", 0.31)
    testevent.setvalue("EventLon", 116.346892)
    testevent.setvalue("EventLonStd", 0.32)
    testevent.setvalue("EventHt", 109.6105)
    testevent.setvalue("EventHtStd", 15)
    testevent.setvalue("CloseRadius", 152)
    testevent.setvalue("FarRadius", 153)

    testevent.setvalue("EventLat2", -33.790603)
    testevent.setvalue("EventLat2Std", 0.33)
    testevent.setvalue("EventLon2", 115.533457)
    testevent.setvalue("EventLon2Std", 0.34)
    testevent.setvalue("EventHt2", 72.9885)
    testevent.setvalue("EventHt2Std", 0.35)

    testuuid = "27e4a2d7-4111-4a72-8a30-969f71fc9207"
    testevent.setvalue("uuid", testuuid)

    return testevent

def createatestevent03():

    testevent = EventContainer("", 0, 0, 0)
    testevent.setvalue("EventTime", "20230525_190849")
    testevent.setvalue("TimeTolerance", 60)
    testevent.setvalue("EventLat", -33.558046)
    testevent.setvalue("EventLatStd", 0.31)
    testevent.setvalue("EventLon", 114.99615)
    testevent.setvalue("EventLonStd", 0.32)
    testevent.setvalue("EventHt", 92.2191)
    testevent.setvalue("EventHtStd", 15)
    testevent.setvalue("CloseRadius", 152)
    testevent.setvalue("FarRadius", 153)

    testevent.setvalue("EventLat2", -33.564671)
    testevent.setvalue("EventLat2Std", 0.33)
    testevent.setvalue("EventLon2", 115.039056)
    testevent.setvalue("EventLon2Std", 0.34)
    testevent.setvalue("EventHt2", 79.9325)
    testevent.setvalue("EventHt2Std", 0.35)

    testuuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    testevent.setvalue("uuid", testuuid)

    return testevent

def createatestevent04():

    testevent = EventContainer("", 0, 0, 0)
    testevent.setvalue("EventTime", "20230525_190849")
    testevent.setvalue("TimeTolerance", 60)
    testevent.setvalue("EventLat", -33.558046)
    testevent.setvalue("EventLatStd", 0.31)
    testevent.setvalue("EventLon", 114.99615)
    testevent.setvalue("EventLonStd", 0.32)
    testevent.setvalue("EventHt", 92.2191)
    testevent.setvalue("EventHtStd", 15)
    testevent.setvalue("CloseRadius", 152)
    testevent.setvalue("FarRadius", 153)

    testevent.setvalue("EventLat2", -33.564671)
    testevent.setvalue("EventLat2Std", 0.33)
    testevent.setvalue("EventLon2", 115.039056)
    testevent.setvalue("EventLon2Std", 0.34)
    testevent.setvalue("EventHt2", 79.9325)
    testevent.setvalue("EventHt2Std", 0.35)

    testuuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    testevent.setvalue("uuid", testuuid)

    return testevent


def createatestevent05():

    testevent = EventContainer("", 0, 0, 0)
    testevent.setvalue("EventTime", "20230517_111327")
    testevent.setvalue("TimeTolerance", 60)
    testevent.setvalue("EventLat", -33.478619)
    testevent.setvalue("EventLatStd", 0.31)
    testevent.setvalue("EventLon", 115.785164)
    testevent.setvalue("EventLonStd", 0.32)
    testevent.setvalue("EventHt", 97.5045)
    testevent.setvalue("EventHtStd", 15)
    testevent.setvalue("CloseRadius", 152)
    testevent.setvalue("FarRadius", 153)

    testevent.setvalue("EventLat2", -32.694214)
    testevent.setvalue("EventLat2Std", 0.33)
    testevent.setvalue("EventLon2", 115.708743)
    testevent.setvalue("EventLon2Std", 0.34)
    testevent.setvalue("EventHt2", 81.0418)
    testevent.setvalue("EventHt2Std", 0.35)

    testuuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    testevent.setvalue("uuid", testuuid)

    return testevent

def createatestevent06():

    testevent = EventContainer("", 0, 0, 0)
    testevent.setvalue("EventTime", "20230526_115827")
    testevent.setvalue("TimeTolerance", 60)
    testevent.setvalue("EventLat", -31.797397)
    testevent.setvalue("EventLatStd", 0.31)
    testevent.setvalue("EventLon", 118.132762)
    testevent.setvalue("EventLonStd", 0.32)
    testevent.setvalue("EventHt", 94.2331)
    testevent.setvalue("EventHtStd", 15)
    testevent.setvalue("CloseRadius", 152)
    testevent.setvalue("FarRadius", 153)

    testevent.setvalue("EventLat2", -32.657057)
    testevent.setvalue("EventLat2Std", 0.33)
    testevent.setvalue("EventLon2", 118.349618)
    testevent.setvalue("EventLon2Std", 0.34)
    testevent.setvalue("EventHt2", 69.2098)
    testevent.setvalue("EventHt2Std", 0.35)

    testuuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    testevent.setvalue("uuid", testuuid)

    return testevent

def createatestevent07():

    testevent = EventContainer("", 0, 0, 0)
    testevent.setvalue("EventTime", "20230526_205441")
    testevent.setvalue("TimeTolerance", 60)
    testevent.setvalue("EventLat", -32.263726)
    testevent.setvalue("EventLatStd", 0.31)
    testevent.setvalue("EventLon", 116.016066)
    testevent.setvalue("EventLonStd", 0.32)
    testevent.setvalue("EventHt", 89.0537)
    testevent.setvalue("EventHtStd", 15)
    testevent.setvalue("CloseRadius", 152)
    testevent.setvalue("FarRadius", 153)

    testevent.setvalue("EventLat2", -32.187818)
    testevent.setvalue("EventLat2Std", 0.33)
    testevent.setvalue("EventLon2", 116.111370)
    testevent.setvalue("EventLon2Std", 0.34)
    testevent.setvalue("EventHt2", 80.8778)
    testevent.setvalue("EventHt2Std", 0.35)

    testuuid = "28e4a2d7-4111-4a72-8a30-969f71fc9207"
    testevent.setvalue("uuid", testuuid)

    return testevent




def testeventcontainer():

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
    testevent02 = createatestevent02()

    if testevent02.eventtostring().split("\n") != expectedresult:
        print("Test event 02 event set values or eventtostring has failed")
        print(expectedresult)
        print(testevent02.eventtostring().split("\n"))
        quit()






def testdatabasefunctions(em):

    operationalconfig = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath(''))


    em.createEventMonitorDB(testmode = True)
    #test calling twice
    em.createEventMonitorDB(testmode = True)
    dbpath = (os.path.expanduser( os.path.join(operationalconfig.data_dir, operationalconfig.event_monitor_db_name)))
    dbfile = pathlib.Path(dbpath)
    if os.path.exists(dbfile):
      dbfile.unlink()
    em.createEventMonitorDB(testmode = True)
    em.addevent(createatestevent02())
    expectedresult = ['# Required ',
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

    eventlist = em.getUnprocessedEventsfromDB()

    match = True
    linecount = 0
    for line in eventlist[0].eventtostring().split("\n"):
     if eventlist[0].eventtostring().split("\n")[linecount] != expectedresult[linecount]:

        print(eventlist[0].eventtostring().split("\n")[linecount])
        print(expectedresult[linecount])
        quit()

     if match:
         print("DB retrieval success")
     else:
         print("DB retrieval fail")
         quit()

    if em.addevent(createatestevent02()):
        print("Quit - added same event twice")
        quit()
    else:
        print("DB rejected duplicate event success")
    pass

    em.markeventasprocessed(eventlist[0])


def testclosestpoint():

    print("Path to platepars {}".format(pathtoplatepars))
    shellcommand = "mkdir -p {} ; \n".format(pathtoplatepars)
    shellcommand += "cd {} ; \n".format(pathtoplatepars)
    shellcommand += plateparstestdata + "  ; \n"
    shellcommand += "tar -xf {}  ; ".format(os.path.join(pathtoplatepars, filename))
    shellcommand += "rm {}       ; ".format(os.path.join(pathtoplatepars, filename))
    print(shellcommand)
    os.system(shellcommand)
    print(os.path.join(pathtoplatepars, "platepars" ,"au0006", ".config"))
    opcon = cr.parse(os.path.join(pathtoplatepars, "platepars", "au0006", ".config"), os.path.abspath('eventmonitor'))
    shellcommand = "cd {} ; ".format(pathtoplatepars)
    shellcommand += "rm -rf platepars"
    em = EventMonitor(opcon)
    em.addevent(createatestevent03())
    events = em.getUnprocessedEventsfromDB()
    e=events[0]

    sd, ed, cd = em.calculateclosestpoint(e.lat,e.lon,e.ht *1000 ,e.lat2,e.lon2,e.ht2 *1000 ,opcon.latitude, opcon.longitude, opcon.elevation)

    # results calculated using http://cosinekitty.com/compass.html
    expectedsd = 179.976# km
    expecteded = 172.718# km

    if abs(expectedsd - sd/1000) > 0.1:
        print("Start distance calculation failed")
        quit()
    else:
        print("Start distance calculation success")

    if abs(expecteded - ed/1000) > 0.1:
        print("End distance calculation failed")
        quit()
    else:
        print("End distance calculation success")

def testtrajectorythroughfovquick(em):

    print("Path to platepars {}".format(pathtoplatepars))
    au0006platepars = os.path.join(pathtoplatepars, "platepars","au0006")
    au0009platepars = os.path.join(pathtoplatepars, "platepars", "au0009")
    au000kplatepars = os.path.join(pathtoplatepars, "platepars", "au000k")
    shellcommand = "cd {} ; ".format(pathtoplatepars)
    shellcommand += "tar -xf {}  ; ".format(os.path.join(pathtoplatepars,filename))
    os.system(shellcommand)
    testrp1 = Platepar()
    print(os.path.join(au0006platepars, "platepar_cmn2010.cal"))
    testrp1.read(os.path.join(au0006platepars, "platepar_cmn2010.cal"))
    opcon = cr.parse(os.path.join(au0006platepars, ".config"), os.path.abspath('eventmonitor'))

    #Test a trajectory that is inside the FoV of AU0006.

    em.addevent(createatestevent03())

    pifov, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1, em.getUnprocessedEventsfromDB()[0])

    if pifov != 100:
        print("FOV calculations failed - AU0006 false negative")
        quit()
    else:
        print("FOV calculation success - AU0006 true positive")

    # AU0009 was colocated with AU0006 but slightly different pointing and did not see this event
    testrp1.read(os.path.join(os.path.expanduser(au0009platepars), "platepar_cmn2010.cal"))
    opcon = cr.parse(os.path.join(os.path.expanduser(au0009platepars), ".config"), os.path.abspath('eventmonitor'))

    pifov, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                                em.getUnprocessedEventsfromDB()[0])

    if pifov != 0:
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

    testrp1.read(os.path.join(os.path.expanduser(au0006platepars), "platepar_cmn2010.cal"))
    opcon = cr.parse(os.path.join(os.path.expanduser(au0006platepars), ".config"), os.path.abspath(''))

    em.addevent(createatestevent05())
    pifov06, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                                  em.getUnprocessedEventsfromDB()[1])

    testrp1.read(
        os.path.join(os.path.expanduser(au0009platepars), "platepar_cmn2010.cal"))
    opcon = cr.parse(os.path.join(os.path.expanduser(au0009platepars), ".config"), os.path.abspath(''))

    em.addevent(createatestevent05())
    pifov09, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                                  em.getUnprocessedEventsfromDB()[1])

    if pifov06 != 100 or pifov09 != 100:
        print("FOV calculations failed - AU0006 or AU0009 false negative")
        quit()
    else:
        print("FOV calculation success - AU0006 and AU0009 true positive")

    # now test a trajectory that is outside both cameras FOV

    # 20230526_115827

    # refer to
    # https://globalmeteornetwork.org/weblog//AU/AU001B/AU001B_20230526_094222_772451_detected/AU001B_20230526_094222_772451_DETECTED_thumbs.jpg

    em.addevent(createatestevent06())

    testrp1.read(
        os.path.join(os.path.expanduser(au0009platepars), "platepar_cmn2010.cal"))
    opcon = cr.parse(os.path.join(os.path.expanduser(au0009platepars), ".config"), os.path.abspath(''))

    pifov09, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                                  em.getUnprocessedEventsfromDB()[2])

    testrp1.read(
        os.path.join(os.path.expanduser(au0006platepars), "platepar_cmn2010.cal"))
    opcon = cr.parse(os.path.join(os.path.expanduser(au0006platepars), ".config"), os.path.abspath(''))


    pifov06, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1, em.getUnprocessedEventsfromDB()[2])

    if pifov06 != 0 or pifov09 != 0:
        print("Fov Calculations failed - AU0006 or AU0009 false postive")
        quit()
    else:
        print("Fov Calculation success - AU0006 and AU0009 true negative")



    # try a non-standard camera AU000K
    # 20230526_205441
    # refer to https://globalmeteornetwork.org/weblog//AU/AU000K/AU000K_20230526_094747_162584_detected/AU000K_20230526_094747_162584_DETECTED_thumbs.jpg

    testrp1.read(
        os.path.join(os.path.expanduser(au000kplatepars), "platepar_cmn2010.cal"))
    opcon = cr.parse(os.path.join(os.path.expanduser(au000kplatepars), ".config"), os.path.abspath(''))

    em.addevent(createatestevent07())
    pifov0K, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1, em.getUnprocessedEventsfromDB()[3])
    if pifov0K != 100:
        print("Fov Calculations failed - AU000K false negative")
        quit()
    else:
        print("Fov Calculation success - AU000K true positive")

    # now fake the camera alt
    testrp1.alt_centre += 10
    pifov0K, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                          em.getUnprocessedEventsfromDB()[3])

    if pifov0K == 100:
        print("Fov Calculations failed - AU000K with a fake offset returned same value")
        print("Safe to ignore this fail, because the Alt and Az are now calculated from the Ra Dec")
    else:
        print("Fov Calculation success - AU000K returned only partial trajectory - {}%".format(pifov0K))

    testrp1.az_centre += 10
    pifov0K, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pifov0K == 100:
        print("Fov Calculations failed - AU000K with a fake offset returned same value")
        print("Safe to ignore this fail, because the Alt and Az are now calculated from the Ra Dec")
    else:
        print("Fov Calculation success - AU000K returned only partial trajectory - {}%".format(pifov0K))

    testrp1.az_centre += 10
    pifov0K, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pifov0K !=0 :
        print("Fov Calculations failed - AU000K returned points when incorrect az")
        print("Safe to ignore this fail, because the Alt and Az are now calculated from the Ra Dec")
    else:
        print("Fov Calculation success - AU000K returned no trajectory visible")

    if pifov0K == 100:
        print("Fov Calculations failed - AU000K with a fake offset returned same value")
        print("Safe to ignore this fail, because the Alt and Az are now calculated from the Ra Dec")
    else:
        print("Fov Calculation success - AU000K returned only partial trajectory - {}%".format(pifov0K))

    testrp1.RA_d  += 10
    pifov0K, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pifov0K == 100:
        print("Fov Calculations failed - AU000K with a fake offset returned same value")
        quit()
    else:
        print("Fov Calculation success - AU000K returned only partial trajectory - {}%".format(pifov0K))

    testrp1.RA_d += 10
    pifov0K, sd, sa, ed, ea, fovra, fovdec = em.trajectoryvisible(testrp1,
                                                                  em.getUnprocessedEventsfromDB()[3])

    if pifov0K != 0:
        print("Fov Calculations failed - AU000K returned points when incorrect RA")
        quit()
    else:
        print("Fov Calculation success - AU000K returned no trajectory visible")

    shellcommand = "cd {} ; ".format(pathtoplatepars)
    shellcommand += "rm -rf platepars"
    print(shellcommand)
    os.system(shellcommand)


def fullsystemtest(config,em):

    """
    Carries a check of the whole system against a snapshot of the GMN database, using a subset of camera information

    Args:
        config: The config for the local system - this is so that the correct file location can be used

    Returns:

    """


    if os.path.isfile(os.path.expanduser("~/RMS_data/test/testlog")):
         os.unlink(os.path.expanduser("~/RMS_data/test/testlog"))

    pathtoplatepars = os.path.expanduser("~/RMS_data/test")
    filename = "archives.tar.bz2"
    print("Path to platepars {}".format(pathtoplatepars))
    shellcommand = "cd {} ; \n".format(pathtoplatepars)

    shellcommand += "wget --quiet http://58.84.202.15:8243/data/archives.tar.bz2 -O archives.tar.bz2 \n"
    shellcommand += "tar -xf {}  ; \n".format(os.path.join(pathtoplatepars,filename))
    print(shellcommand)
    os.system(shellcommand)

    filename = "trajectorydata.tar.bz2"
    shellcommand += "wget --quiet http://58.84.202.15:8243/data/trajectorydata.tar.bz2 -O trajectorydata.tar.bz2 \n"
    shellcommand += "tar -xf {}  ; \n".format(os.path.join(pathtoplatepars,filename))
    print(shellcommand)
    os.system(shellcommand)


    cameradataavailable = os.listdir(os.path.join(pathtoplatepars,"archives"))
    cameradataavailable.sort()
    trajectoryfiles = os.listdir(os.path.expanduser(os.path.join(pathtoplatepars,"trajectorydata")))
    trajectoryfiles.sort()
    trajectorylist = []

    for trajectoryfile in tqdm(trajectoryfiles):

        if trajectoryfile[0:12] != "traj_summary":
            continue
        with open(os.path.join(os.path.join(pathtoplatepars,"trajectorydata/"),trajectoryfile)) as fh:

          filetowrite = False
          for trajectory in fh:
            if trajectory[0] != "#" :

              if trajectory.strip() != "" and trajectory.strip() != "\n":

                trajectorylist.append(trajectory.strip())

    if os.path.exists(os.path.expanduser('~/RMS_data/event_watchlist.txt')):
        os.unlink(os.path.expanduser('~/RMS_data/event_watchlist.txt'))

    if os.path.exists(os.path.expanduser(os.path.join(config.data_dir, "testlog"))):
        os.unlink(os.path.expanduser(os.path.join(config.data_dir, "testlog")))


    logfile = open(os.path.expanduser(os.path.join(config.data_dir, "testlog")), 'wt')
    for traj in tqdm(trajectorylist):
        relevanttrajectory = False
        #create a fresh event_watchlist.txt page
        #need to open as write so that we overwrite any previous trajectories


        t = traj.split(';')
        eventtime, trajcam   = t[2] , t[85].strip().split(",")
        latbeg,lonbeg, htbeg = t[63], t[65], t[67]
        latend,lonend, htend = t[69], t[71], t[73]
        time.sleep(0.0001)

        # Check 1 - are observations that were seen predicted correctly
        # Iterate through all the cameras GMN associated with the event
        for camera in trajcam:
                    # If any single camera in that event is in the local database
                    if camera.lower() in cameradataavailable:
                        # And we have not already written an even_watchlist for this trajectory
                        if not relevanttrajectory:
                               #write all the cameras that saw the event to log file


                               with open(os.path.expanduser(os.path.join(em.syscon.data_dir, "testlog")), 'at') as logfile:
                                logfile.write("{} GMN {}  \n".format(eventtime.strip().split('.')[0],trajcam))

                               # and write an event_watch.txt file
                               # prepare an event and write to the local file for testing
                               tev = EventContainer(0,0,0,0)
                               tev.dt, tev.timetolerance      = parser.parse(eventtime).strftime("%Y%m%d_%H%M%S"), 5
                               tev.lat, tev.lon, tev.ht       = float(latbeg), float(lonbeg), float(htbeg)
                               tev.lat2, tev.lon2, tev.ht2    = float(latend), float(lonend), float(htend)
                               tev.closeradius, tev.farradius = 1, 1000

                               with open(os.path.expanduser('~/RMS_data/event_watchlist.txt'), 'wt') as watchfh:
                                 watchfh.write(tev.eventtostring() + "\n")

                                 relevanttrajectory = True

        # For every camera, we need to check whether it saw the trajectory
        # This can't be a subsection of main loop, otherwise we only check for cameras that were used in the solver
        for camera in cameradataavailable:

         if relevanttrajectory:
          print(camera)
          evcon = cr.Config()
          evcon.data_dir = os.path.join(pathtoplatepars,"archives",camera.lower(),"RMS_data")
          em.config.data_dir = os.path.join(pathtoplatepars,"archives",camera.lower(),"RMS_data")
          #and write the camera name
          evcon.stationID = camera.upper()
          # set the system config in eventmonitor
          em.syscon.data_dir = "~/RMS_data"
          em.geteventsandcheck(testmode=True)

        unprocessedevents = em.getUnprocessedEventsfromDB()
        # because we are in testmode, the events do not get marked as processed,
        for unprocessedevent in unprocessedevents:
            em.markeventasprocessed(unprocessedevent)







    shellcommand = "cd {} ; ".format(pathtoplatepars)
    os.system(shellcommand)


def testindividuals():



    if convertgmntimetoposix("20210925_163127") == datetime(2021,9,25,16,31,27):
        print("convertgmntimetoposix success")
    else:
        print("convertgmntimetoposix fail")
        quit()

def functiontest(config,em):

    testindividuals()
    testeventcontainer()
    testdatabasefunctions(em)
    testclosestpoint()
    testtrajectorythroughfovquick(em)
    fullsystemtest(config,em)
    quit()

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="""Test procedure for Event Monitor. \
        """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-c', '--config', nargs=1, metavar='CONFIG_PATH', type=str, \
                            help="Path to a config file which will be used instead of the default one.")

    cml_args = arg_parser.parse_args()

    config = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('eventmonitor'))
    em = EventMonitor(config)
    em.syscon = cr.loadConfigFromDirectory(cml_args.config, os.path.abspath('eventmonitor'))


    functiontest(config,em)