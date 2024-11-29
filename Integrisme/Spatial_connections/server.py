import http.server
import socketserver

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.js'):
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            with open('.' + self.path, 'rb') as file:
                self.wfile.write(file.read())
        else:
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

PORT = 8000
Handler = MyHttpRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()