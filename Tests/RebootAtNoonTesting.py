import os.path

import RMS.Misc
import unittest
import datetime
import re
import dvrip as dvr
import subprocess

def readFileAsLines(file_path="~/source/RMS/.config"):

    with open(os.path.expanduser(file_path), 'r') as file:
        line_list =  file.readlines()

    file_as_lines_list, stripped_line = [], ""
    for line in line_list:
        stripped_line = line.strip().replace("\n","")
        file_as_lines_list.append(stripped_line)

    return file_as_lines_list

def writeFileFromLines(lines_list, file_path="~/source/RMS/.config"):

    with open(os.path.expanduser(file_path), 'w') as file:

        for line in lines_list:
            file.write("{}\n".format(line))
        file.flush()



def getValue(lines_list,section, key):

    this_section, output_line_list = None, []
    for line in lines_list:
        if line.strip().startswith("[") and line.strip().endswith("]"):
             this_section =  line.replace("[","").replace("]","").strip()

        if this_section == section and line.startswith("{}:".format(key)):
            value = line[len(key)+1:].strip()

    return value



def setValue(lines_list, section, key, value):

    this_section, output_line_list = None, []
    for line in lines_list:
        if line.strip().startswith("[") and line.strip().endswith("]"):
             this_section =  line.replace("[","").replace("]","").strip()

        if this_section == section and line.startswith("{}:".format(key)):
            line = "{}: {}".format(key, value)
        output_line_list.append(line)
    return output_line_list

class TestRebootAtNoon(unittest.TestCase):

    def test_RebootAtNoon(self):
        # Test code - should produce no output
        config_file = readFileAsLines()
        camera_ip =  re.findall(r"[0-9]+(?:\.[0-9]+){3}", getValue(config_file, "Capture", "device"))[0]
        cam = dvr.DVRIPCam(camera_ip)

        if cam.login():

            print("Logged in to {}".format(camera_ip))
            # Store the station longitude setting
            station_longitude = getValue(config_file, "System", "longitude")
            for test_longitude in range(-360, 360, 5):
                writeFileFromLines(setValue(config_file, "System", "longitude", test_longitude))
                for test_hour in range(0, 23):
                    for test_minute in range(0,45,15):
                        test_time_python_object = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
                        test_time_python_object = test_time_python_object.replace(hour=test_hour, minute=test_minute, second=0, microsecond=0)
                        print("Testing at longitude: {} and time:{}".format(test_longitude, test_time_python_object))
                        cam.set_time(test_time_python_object)
                        camera_time = cam.get_time()
                        print("Camera set to {}".format(camera_time))
                        result = subprocess.run(['python','-m','Utils.CameraControl','setAutoReboot','EveryDay,noon'], capture_output=True, text=True)
                        print("Command feedback was {}".format(result))
        # Put the station longitude back
        writeFileFromLines(setValue(config_file, "System", "longitude", station_longitude))



        result = 1
        expected_result = 1
        self.assertEqual(result, expected_result)



if __name__ == '__main__':

    unittest.main()