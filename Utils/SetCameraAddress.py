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
    """ Check if a given IP address is valid. All subnets need to be in the 1-254 range. """
    
    spls = addr.split('.')
    
    if len(spls) != 4: 
        return False
    
    try:
        i0 = int(spls[0])
        i1 = int(spls[1])
        i2 = int(spls[2])
        i3 = int(spls[3])
    except:
        return False
    
    ip_range = list(range(1, 255))
    if (i0 not in ip_range) or (i1 > 254) or (i2 > 254) or (i3 not in ip_range):
        return False
    
    return True
    

def strIPtoHex(ip):
    
    a = binascii.hexlify(mysocket.inet_aton(ip)).decode().upper()
    
    addr = '0x' + ''.join([a[x:x + 2] for x in range(0, len(a), 2)][::-1])
    
    return addr


def iptoString(s):
    a = s[2:]
    addr = '0x'+''.join([a[x:x + 2] for x in range(0, len(a), 2)][::-1])
    ipaddr = ip.IPv4Address(int(addr, 16))
    
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
        if sys.version_info[0] > 2:
            md5 = hashlib.md5(bytes(password, "utf-8")).digest()
        else:
            md5 = hashlib.md5(password.decode('utf-8')).digest()

        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        return "".join([chars[sum(x) % 62] for x in zip(md5[::2], md5[1::2])])

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


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("This script allows you to set your Camera's IP address.")
        print('')
        print('To use the script, your camera must be attached to your home network')
        print('and acessible via its default IP address. If you are not sure what')
        print('what that is, you may be able to find out from your home router. Most routers')
        print("have a page that displys connected clients. Look for one named 'HostName',")
        print('and make a note of its address. Alternatively you can use free tools like ')
        print('Advanced IP-Scanner to scan your network for the same name. ')
        print('')
        print('You will also need to know the address you want your camera to have.')
        print('Normally this will be 192.168.42.10')
        print('')
        print('Once you have this information and the camera is on your network')
        print('call this module again with two arguments, the CURRENT and DESIRED address, eg:')
        print('')
        print('   python -m Utils.SetCameraAddress 192.168.1.100 192.168.42.10')
        print('')
        print('replacing the two addresses as needed')
        print('')
        exit(0)

    ipaddr = sys.argv[1]
    newaddr = sys.argv[2]

    if not checkValidIPAddr(ipaddr) or not checkValidIPAddr(newaddr):
        print('')
        print('One or both IP Addresses seems invalid - check that they are correct and in ')
        print('dotted form ie a.b.c.d where a,b,c and d are numbers between 1 and 254')
        print('')
        print('')
        exit(0)

    cam=DVRIPCam(ipaddr)
    if cam.login():

        print('--------')
        print('The process will appear to hang for several seconds then should print the ')
        print('new address. ')
        print('--------')
        nc=cam.get_info("NetWork.NetCommon.HostIP")
        dh=cam.get_info("NetWork.NetDHCP.[0].Enable")
        print('current address {}, dhcp enabled is {}'.format(iptoString(nc), dh))

        print('--------')
        print('setting address to {}'.format(newaddr))
        print('--------')
        cam.set_info("NetWork.NetDHCP.[0].Enable", 0)
        hexval = strIPtoHex(newaddr)
        try: 
            # this wil actually succeed, but a timeout will occur once
            # the camera address is changed.
            cam.set_info("NetWork.NetCommon.HostIP", hexval)
        except mysocket.timeout:
            cam2=DVRIPCam(newaddr)
            cam2.login()
            nc=cam2.get_info("NetWork.NetCommon.HostIP")
            dh=cam2.get_info("NetWork.NetDHCP.[0].Enable")
            print('Address now {}, dhcp enabled is {}'.format(iptoString(nc), dh))
            cam2.close()
            cam.close()
            print('--------')
            exit(0)
    else:
        print('login failed')
