import socket


# Create a class to manage the socket connection
class ServerManager:
    ip = "127.0.0.1"
    port = 8888
    socket = None

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))

    def send(self, action):
        self.socket.send(action.encode())

    def receive(self):
        return self.socket.recv(1024).decode()

    def close(self):
        self.socket.close()
