# class that represent the unity's environment with actions to send to unity and receive the state and reward from unity
# the class can communicate with the unity environment through the server_manager class (send/receive)
from server_manager import ServerManager
from enum import Enum

class Environment:
    def __init__(self):
        self.server_manager = ServerManager()
        self.state = None
        self.reward = None

    class Action(Enum):
        MOVE_UP = 0,
        MOVE_RiGHT = 1,
        MOVE_DOWN = 2,
        MOVE_LEFT = 3,
        TURN_RIGHT = 4,
        TURN_LEFT = 5,
        JUMP = 6,
    
        @staticmethod
        def get_name_from_value(value):
            for name, member in Environment.Action.__members__.items():
                if member.value == value:
                    return name
            
    def start(self):
        self.server_manager.start()
        self.reset()
            
    def stop(self):
        self.server_manager.stop()
            
    def reset(self):
        self.server_manager.send("reset")

    def set_action(self, action):
        action = Environment.Action.get_name_from_value(action)
        self.server_manager.send("set_action:" + str(action))
        message = self.server_manager.receive()
        #parse the reward, state and done from the message separated by :
        self.reward, self.state, done = message.split(":") # reward (float), state (int), done (boolean)
        return self.reward, self.state, done

