#!/usr/bin/env python3
"""Simple HTTP server to serve the frontend."""

import http.server
import socketserver
import os

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"✅ Server running at http://localhost:{PORT}")
    print(f"📂 Serving from: {os.getcwd()}")
    print(f"🌐 Open http://localhost:{PORT}/index.html in your browser")
    print("\nPress Ctrl+C to stop the server\n")
    httpd.serve_forever()
