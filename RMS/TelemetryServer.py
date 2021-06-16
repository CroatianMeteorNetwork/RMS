""" Generic Telemetry web service """

from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
from socketserver import BaseRequestHandler
import time
import threading
import re
import cv2
from typing import Callable, Tuple

class TelemetryServer(ThreadingHTTPServer):
    data = {}
    data_frame = None

    def __init__(self, ip, port):
        super().__init__((ip, port), TelemetryHandler)
        

    def set_data(self, data_obj):
        self.data = data_obj

    def set_data_frame(self, data_frame_obj):
        self.data_frame = data_frame_obj

    def run(self):
        server_thread = threading.Thread(target=self.serve_forever)
        server_thread.daemon_threads = True
        server_thread.start()

class  TelemetryHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        m = re.search('GET (/.*) HTTP.+', self.requestline)
        if m:
            self.handle_request(m.group(1))
        else:
            self.send_response(500)
            self.wfile.write('Internal Error - parsing requestline')

    def handle_request(self, req):

        if req == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            self.wfile.write(bytes(json.dumps(self.server.data), "utf-8"))
            self.wfile.flush()
        elif req == '/last_frame':
            if self.server.data_frame is None:
                self.send_response(440)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(bytes('No data', 'utf-8'))
                self.wfile.flush()
            else:
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()

                _, frame = cv2.imencode('.JPEG', self.server.data_frame)

                self.wfile.write(frame)
                self.wfile.flush()

        else:
            self.send_response(440)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()

            self.wfile.write(bytes('Not found', 'utf-8'))
            self.wfile.flush()




if __name__ == "__main__":
    server = TelemetryServer('127.0.0.1', 5000)
    server.run()

    print("server started")
    server.set_data({'hello': 'world', 'received': 'ok'})


    input("Press any key to terminate the program")
    server.set_data({'hello': 'world', 'received': 'err'})
    input("Press any key to terminate the program")

    server.shutdown() 
    print("Server stopped.")        

