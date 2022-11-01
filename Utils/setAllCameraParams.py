import struct
import json
import hashlib
import threading
from socket import socket, AF_INET, SOCK_STREAM, SOCK_DGRAM
import time
import logging
import sys
import binascii
import ipaddress as ip
import socket as mysocket


def checkValidIPAddr(addr):
    spls = addr.split('.')
    if len(spls) != 4: 
        return False
    for s in spls:
        try:
            intv = int(s)
        except:
            return False
        if intv < 1 or intv > 254:
            return False
    return True
    

def strIPtoHex(ip):
    a = binascii.hexlify(mysocket.inet_aton(ip)).decode().upper()
    addr='0x'+''.join([a[x:x+2] for x in range(0,len(a),2)][::-1])
    return addr


def iptoString(s):
    a=s[2:]
    addr='0x'+''.join([a[x:x+2] for x in range(0,len(a),2)][::-1])
    ipaddr=ip.IPv4Address(int(addr,16))
    return ipaddr


class DVRIPCam(object):
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    OK_CODES = [100, 515]
    PORTS = {
        "tcp": 34567,
        "udp": 34568,
    }

    def __init__(self, ip, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.ip = ip
        self.user = kwargs.get("user", "admin")
        hash_pass = kwargs.get("hash_pass")
        self.hash_pass = hash_pass or self.sofia_hash(kwargs.get("password", ""))
        self.proto = kwargs.get("proto", "tcp")
        self.port = kwargs.get("port", self.PORTS.get(self.proto))
        self.socket = None
        self.packet_count = 0
        self.session = 0
        self.alive_time = 20
        self.alive = None
        self.alarm = None
        self.alarm_func = None
        self.busy = threading.Condition()

    def connect(self, timeout=2):
        if self.proto == "tcp":
            self.socket_send = self.tcp_socket_send
            self.socket_recv = self.tcp_socket_recv
            self.socket = socket(AF_INET, SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
        elif self.proto == "udp":
            self.socket_send = self.udp_socket_send
            self.socket_recv = self.udp_socket_recv
            self.socket = socket(AF_INET, SOCK_DGRAM)
        else:
            raise 'Unsupported protocol {}'.format(self.proto)

        self.timeout = timeout
        self.socket.settimeout(timeout)

    def close(self):
        self.alive.cancel()
        self.socket.close()
        self.socket = None

    def udp_socket_send(self, bytes):
        return self.socket.sendto(bytes, (self.ip, self.port))

    def udp_socket_recv(self, bytes):
        data, _ = self.socket.recvfrom(bytes)
        return data

    def tcp_socket_send(self, bytes):
        return self.socket.sendall(bytes)

    def tcp_socket_recv(self, bufsize):
        return self.socket.recv(bufsize)

    def receive_with_timeout(self, length):
        received = 0
        buf = bytearray()
        start_time = time.time()

        while True:
            data = self.socket_recv(length - received)
            buf.extend(data)
            received += len(data)
            if length == received:
                break
            elapsed_time = time.time() - start_time
            if elapsed_time > self.timeout:
                return None
        return buf

    def receive_json(self, length):
        data = self.receive_with_timeout(length).decode('utf-8')
        if data is None:
            return {}

        self.packet_count += 1
        self.logger.debug("<= %s", data)
        reply = json.loads(data[:-2])
        return reply

    def send(self, msg, data={}, wait_response=True):
        if self.socket is None:
            return {"Ret": 101}
        # self.busy.wait()
        self.busy.acquire()
        if hasattr(data, "__iter__"):
            if sys.version_info[0] > 2:
                data = bytes(json.dumps(data, ensure_ascii=False), "utf-8")
            else:
                data = json.dumps(data, ensure_ascii=False)
        pkt = (
            struct.pack(
                "BB2xII2xHI",
                255,
                0,
                self.session,
                self.packet_count,
                msg,
                len(data) + 2,
            )
            + data
            + b"\x0a\x00"
        )
        self.logger.debug("=> %s", pkt)
        self.socket_send(pkt)
        if wait_response:
            reply = {"Ret": 101}
            (
                head,
                version,
                self.session,
                sequence_number,
                msgid,
                len_data,
            ) = struct.unpack("BB2xII2xHI", self.socket_recv(20))
            reply = self.receive_json(len_data)
            self.busy.release()
            return reply

    def sofia_hash(self, password=""):
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        if sys.version_info[0] > 2:
            md5 = hashlib.md5(bytes(password, "utf-8")).digest()
            return "".join([chars[sum(x) % 62] for x in zip(md5[::2], md5[1::2])])
        else:
            md5 = hashlib.md5(password.decode('utf-8')).digest()
            iterlist = zip(md5[::2], md5[1::2])
            cc = []
            for x in iterlist:
                su=0
                for xx in x:
                    su = su + ord(xx)
                cc.append(chars[su % 62])
            return "".join(cc)

    def login(self):
        if self.socket is None:
            self.connect()
        data = self.send(
            1000,
            {
                "EncryptType": "MD5",
                "LoginType": "DVRIP-Web",
                "PassWord": self.hash_pass,
                "UserName": self.user,
            },
        )
        if data["Ret"] not in self.OK_CODES:
            return False
        self.session = int(data["SessionID"], 16)
        self.alive_time = data["AliveInterval"]
        self.keep_alive()
        return data["Ret"] in self.OK_CODES

    def keep_alive(self):
        self.send(
            1006,
            {"Name": "KeepAlive", "SessionID": "0x%08X" % self.session},
        )
        self.alive = threading.Timer(self.alive_time, self.keep_alive)
        self.alive.start()

    def get_info(self, command):
        return self.get_command(command, 1042)
    
    def get_command(self, command, code):
        data = self.send(code, {"Name": command, "SessionID": "0x%08X" % self.session})
        if data["Ret"] in self.OK_CODES and command in data:
            return data[command]
        else:
            return data

    def set_info(self, command, data):
        return self.set_command(command, data, 1040)

    def set_command(self, command, data, code):
        return self.send(
            code, {"Name": command, "SessionID": "0x%08X" % self.session, command: data}
        )


def setCameraParam(cam, opts):
    """ Set a parameter in the Camera section of the camera config
        Individual parameters can be set and the change will take effect immediately 
    Args:
        cam - the camera 
        opts - array of fields, subfields and the value to set
    """
    # these fields are stored as integers. Others are Hex strings
    intfields=['AeSensitivity','Day_nfLevel','DncThr','ElecLevel',
        'IRCUTMode','IrcutSwap','Night_nfLevel', 'Level','AutoGain','Gain']
    styleFlds='typedefault','type1','type2'

    # top level field name
    fld=opts[1]
    # these fields are stored in the ParamEx.[0] block
    if fld == 'Style':
        val = opts[2]
        if val not in styleFlds:
            print('style must be one of ', styleFlds)
            return
        print('Set Camera.ParamEx.[0].{} to {}'.format(fld, val))
        cam.set_info("Camera.ParamEx.[0]",{fld:val})
    elif fld == 'BroadTrends': 
        subfld=opts[2]
        val = int(opts[3])
        fldToSet='Camera.ParamEx.[0].' + fld 
        print('Set {}.{} to {}'.format(fldToSet, subfld, val))
        cam.set_info(fldToSet,{subfld:val})
                
    # Exposuretime and gainparam have subfields
    elif fld == 'ExposureParam' or fld == 'GainParam':
        subfld=opts[2]
        val = int(opts[3])
        if subfld not in intfields:
            # the two non-int fields in ExposureParam are the exposure times. 
            # These are stored in microsconds converted ito hex strings.
            if val < 100 or val > 80000: 
                print('Exposure must be between 100 and 80000 microsecs')
                return
            val ="0x%8.8X" % (int(val))
        fldToSet='Camera.Param.[0].' + fld 
        print('Set {}.{} to {}'.format(fldToSet, subfld, val))
        cam.set_info(fldToSet,{subfld:val})
    else:
        # other fields do not have subfields
        val = int(opts[2])
        if fld not in intfields:
            val ="0x%8.8X" % val
        print('Set Camera.Param.[0].{} to {}'.format(fld, val))
        cam.set_info("Camera.Param.[0]",{fld:val})


if __name__ == '__main__':

    ipaddr = sys.argv[1]

    if not checkValidIPAddr(ipaddr):
        print('ipaddress invalid')
        print('')
        exit(0)

    cam=DVRIPCam(ipaddr)
    if cam.login():

        nc=cam.get_info("NetWork.NetCommon.HostIP")
        dh=cam.get_info("NetWork.NetDHCP.[0].Enable")
        print('current address {}, dhcp enabled is {}'.format(iptoString(nc), dh))

        info = cam.get_info("AVEnc.VideoColor.[0]")
        b,c,s,h,g,a = 100,50,0,50,0,0
        n = 0
        info[n]["VideoColorParam"]["Brightness"] = b
        info[n]["VideoColorParam"]["Contrast"] = c
        info[n]["VideoColorParam"]["Saturation"] = s
        info[n]["VideoColorParam"]["Hue"] = h
        info[n]["VideoColorParam"]["Gain"] = g
        info[n]["VideoColorParam"]["Acutance"] = a
        print('Set color configuration', b,c,s,h,g,a)
        cam.set_info("AVEnc.VideoColor.[0]", info)

        info = cam.get_info("AVEnc.VideoWidget")
        info[0]["TimeTitleAttribute"]["EncodeBlend"] = False 
        info[0]["ChannelTitleAttribute"]["EncodeBlend"] = False 
        print('Set osd disabled')
        cam.set_info("AVEnc.VideoWidget", info)

        params = cam.get_info("Simplify.Encode")
        params[0]['MainFormat']['Video']['Compression'] = 'H.264'
        params[0]['MainFormat']['Video']['Resolution'] = '720P'
        params[0]['MainFormat']['Video']['BitRateControl'] = 'VBR'
        params[0]['MainFormat']['Video']['FPS'] = 25
        params[0]['MainFormat']['Video']['Quality'] = 6

        params[0]['MainFormat']['VideoEnable']=1
        params[0]['MainFormat']['AudioEnable']=0
        params[0]['ExtraFormat']['VideoEnable']=0
        params[0]['ExtraFormat']['AudioEnable']=0
        cam.set_info("Simplify.Encode", params)

        setCameraParam(cam, ['','Style','type1'])
        setCameraParam(cam, ['','AeSensitivity','1'])
        setCameraParam(cam, ['','ApertureMode','0'])
        setCameraParam(cam, ['','BLCMode','0'])
        setCameraParam(cam, ['','DayNightColor','2'])
        setCameraParam(cam, ['','Day_nfLevel','0'])
        setCameraParam(cam, ['','DncThr','50'])
        setCameraParam(cam, ['','ElecLevel','100'])
        setCameraParam(cam, ['','EsShutter','0'])
        setCameraParam(cam, ['','IRCUTMode','0'])
        setCameraParam(cam, ['','IrcutSwap','0'])
        setCameraParam(cam, ['','Night_nfLevel','0'])
        setCameraParam(cam, ['','RejectFlicker','0'])
        setCameraParam(cam, ['','WhiteBalance','2'])
        setCameraParam(cam, ['','PictureFlip','0'])
        setCameraParam(cam, ['','PictureMirror','0'])
        setCameraParam(cam, ['','ExposureParam','LeastTime','40000'])
        setCameraParam(cam, ['','ExposureParam','MostTime','40000'])
        setCameraParam(cam, ['','ExposureParam','Level','0'])
        setCameraParam(cam, ['','GainParam','AutoGain','1'])
        setCameraParam(cam, ['','GainParam','Gain','70'])


        cam.close()
        print('--------')
        exit(0)
    else:
        print('login failed')
