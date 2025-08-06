import os.path
import json
import RMS.Misc
import unittest
import datetime
import re
import dvrip as dvr
import subprocess
import pickle

process_pickle_only = True

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

        config_file = readFileAsLines()
        rms_data_path = os.path.expanduser(getValue(config_file, "Capture", "data_dir"))

        if not process_pickle_only:
            camera_ip = re.findall(r"[0-9]+(?:\.[0-9]+){3}", getValue(config_file, "Capture", "device"))[0]
            cam = dvr.DVRIPCam(camera_ip)

            if cam.login():

                print("Logged in to {}".format(camera_ip))
                # Store the station longitude setting
                station_longitude = getValue(config_file, "System", "longitude")
                test_data = []
                for test_longitude in range(-360, 360, 10):
                    writeFileFromLines(setValue(config_file, "System", "longitude", test_longitude))
                    for test_hour in range(0, 23):
                        for test_minute in range(0,60,15):
                            real_time = datetime.datetime.now(tz=datetime.timezone.utc).replace(tzinfo=None)
                            test_time_python_object = real_time.replace(hour=test_hour, minute=test_minute, second=0, microsecond=0)
                            print("Testing at longitude: {} and time:{}".format(test_longitude, test_time_python_object))
                            cam.set_time(test_time_python_object)
                            camera_time = cam.get_time()
                            print("Camera set to {}".format(camera_time))
                            result = subprocess.run(['python','-m','Utils.CameraControl','SetAutoReboot','Everyday,noon'], capture_output=True, text=True)
                            print("Command feedback was {}".format(result))
                            reboot_hour_read_back = cam.get_info("General.AutoMaintain")['AutoRebootHour']
                            print("Reboot time read back as {}".format(reboot_hour_read_back))
                            test_data.append([test_longitude,real_time,camera_time,reboot_hour_read_back])
                with open(os.path.join(rms_data_path,"reboot_at_noon_test.pkl"), 'wb') as f:
                    pickle.dump(test_data, f)

            # Put the station longitude back
                writeFileFromLines(setValue(config_file, "System", "longitude", station_longitude))

        with open(os.path.join(rms_data_path, "reboot_at_noon_test.pkl"), 'rb') as f:
            test_data_list = pickle.load(f)
        output_data = []
        max_divergence = 0
        output_data.append(
            "Longitude | Longitude wrapped | Local noon | Test time | Camera time | Camera time ahead (hrs) | Noon in camera time | Camera reboot time | Divergence |\n")
        for test_data in test_data_list:
            longitude, test_time_utc, camera_time, reboot_hour_read_back = test_data
            longitude_wrapped = (longitude + 180) % 360 - 180




            local_noon_utc = 12 - 24 * longitude_wrapped / 360
            camera_time_offset_from_utc_hours = (camera_time - test_time_utc).total_seconds() / 3600
            noon_in_camera_time = (local_noon_utc + camera_time_offset_from_utc_hours) % 24
            divergence_hrs = round(min(noon_in_camera_time - reboot_hour_read_back, reboot_hour_read_back, noon_in_camera_time),1)
            output_data.append("{:>9} |{:>18} | {:>10} | {:>5}  | {:>5}    | {:>23} | {:>19} | {:>18} | {:>10} |\n".format(longitude, longitude_wrapped,
                                    round(local_noon_utc,1), test_time_utc.strftime("%H:%M:%S"),
                                    camera_time.strftime("%H:%M:%S"), round(camera_time_offset_from_utc_hours,1),
                                    round(noon_in_camera_time,1), reboot_hour_read_back, divergence_hrs))

            pass

            self.assertLess (divergence_hrs, 2 )

        with open(os.path.join(rms_data_path, "reboot_at_noon_test.txt"), 'w') as f:
            f.writelines(output_data)
            f.flush()

if __name__ == '__main__':

    unittest.main()