from http.server import HTTPServer, CGIHTTPRequestHandler
import os

# Make sure the server is created at current directory
os.chdir('.')
print("Open https://localhost:8080/robot.html")

server = HTTPServer(('', 8080), RequestHandlerClass=CGIHTTPRequestHandler)
server.serve_forever()