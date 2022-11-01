""" Controls ONVIF-compatible IP camera """

# Prerequisites:
# Python2 
#   pip install onvif
# python3
#   pip install onvif_zeep
# note that if you install onvif_zeep on python2 it will not work

# Usage examples:
#
# python -m Utils.CameraControl27 reboot
# python -m Utils.CameraControl27 getDeviceInformation

from __future__ import print_function

import logging
import threading
from socket import socket, AF_INET, SOCK_STREAM, SOCK_DGRAM
import json
import time 
import sys 
import hashlib
import struct

if sys.version_info.major > 2:
    print('use Utils.CameraControl with Python3')
    exit(0)


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
            md5=struct.unpack('>BBBBBBBBBBBBBBBB', md5)

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
