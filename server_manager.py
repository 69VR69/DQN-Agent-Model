import socket
import re
import numpy as np

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
        print("connecting to server ("+self.ip + "/" + str(self.port) + ")...")
        self.socket.connect((self.ip, self.port))
        print("connected to server")

    def is_running(self):
        return self.socket is not None

    def send(self, message):
        if self.is_response_expected:
            print("still waiting for response...")
            return
        print("sending : " + message)
        self.socket.send(message.encode() + b"\n")
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

class ServerManagerMock(ServerManager):
    def __init__(self):
        self.socket = None
        self.is_response_expected = False
        self.ip = "127.0.0.1"
        self.port = 8888
        self.received_message = 0
    
    def start(self):
        print("server starting...")
        print("connecting to server ("+self.ip + "/" + str(self.port) + ")...")
        print("connected to server")
        self.socket = 1

    def send(self, message):
        if self.is_response_expected:
            print("still waiting for response...")
            return
        #print("sending : " + message)
        if(message == "reset"):
            self.received_message = 0
        elif(re.search("set_action", message)):
            self.received_message = 1
        elif(message == "get_state"):
            self.received_message = 2
        self.is_response_expected = True
    
    def receive(self):
        if(self.received_message == 0):
            message = "0:0,0,0,-1,0,0,-1,0,0;0,0,0,0,1,0,0,0,0;0,0,0,0,0,0,0,0,0;:0"
        elif(self.received_message == 1):
            message = "0:0,0,0,-1,0,0,-1,0,0;0,0,0,0,1,0,0,0,0;0,0,0,0,0,0,0,0,0;:0"
        elif(self.received_message == 2):
            message = "0,0,0,-1,0,0,-1,0,0;0,0,0,0,1,0,0,0,0;0,0,0,0,0,0,0,0,0;"
        self.is_response_expected = False
        #print("received : " + message)
        return message
    
    def close(self):
        print("closing connection...")
        print("connection closed")