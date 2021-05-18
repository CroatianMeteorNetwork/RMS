""" Generic Telemetry web service """

from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
import json
from socketserver import BaseRequestHandler
import time
import threading
from typing import Callable, Tuple

class TelemetryServer(ThreadingHTTPServer):
    data = {}

    def __init__(self, ip, port):
        super().__init__((ip, port), TelemetryHandler)
        

    def set_data(self, data_obj):
        self.data = data_obj

    def run(self):
        server_thread = threading.Thread(target=self.serve_forever)
        server_thread.daemon_threads = True
        server_thread.start()

class  TelemetryHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        self.wfile.write(bytes(json.dumps(self.server.data), "utf-8"))
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

