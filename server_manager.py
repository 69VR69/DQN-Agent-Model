import socket


# Create a class to manage the socket connection
class ServerManager:
    def __init__(self):
        self.socket = None
        self.is_response_expected = False
        self.ip = "127.0.0.1"
        self.port = 8888
    
    def start(self):
        print("server starting...")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10)
        print("connecting to server ("+self.ip + "/" + str(self.port) + ")...")
        self.socket.connect((self.ip, self.port))
        print("connected to server")


    def send(self, message):
        if self.is_response_expected:
            print("still waiting for response...")
            return
        message = message + "|"
        print("sending : " + message)
        self.socket.send(message.encode())
        self.is_response_expected = True

    def receive(self):
        message = self.socket.recv(1024).decode()
        self.is_response_expected = False
        print("received : " + message)
        return message

    def close(self):
        print("closing connection...")
        self.socket.close()
        print("connection closed")
