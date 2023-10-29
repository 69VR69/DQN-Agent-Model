# class that represent the unity's environment with actions to send to unity and receive the state and reward from unity
# the class can communicate with the unity environment through the server_manager class (send/receive)
from server_manager import ServerManager, ServerManagerMock
from enum import Enum
import numpy as np

class Environment:
    def __init__(self):
        self.server_manager = ServerManagerMock()#ServerManager()
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
        WAIT = 7,
    
        @staticmethod
        def get_name_from_value(value):
            # transform int to tuple
            if isinstance(value, int):
                value = (value,)
            
            for name, member in Environment.Action.__members__.items():
                if member.value == value:
                    return name
            
    def start(self):
        self.server_manager.start()

    def is_running(self):
        return self.server_manager.is_running()
            
    def stop(self):
        self.server_manager.stop()
            
    def reset(self):
        self.server_manager.send("reset")
        return self.get_info()[1]

    def set_action(self, action):
        action = Environment.Action.get_name_from_value(action)
        self.server_manager.send("set_action:" + str(action))
        return self.get_info()
    
    def get_state(self):
        self.server_manager.send("get_state")
        message = self.server_manager.receive()
        return self.parse_state(message)
    
    def get_info(self):
        message = self.server_manager.receive()
        return self.parse_message(message)
    
    def parse_message(self, message):
        #parse the reward, state and done from the message separated by :
        self.reward, self.state, done = message.split(":") # reward (float), state (array2d of float), done (boolean)
        self.state = self.parse_state(self.state)
        return float(self.reward), self.state, done == 1
    
    def parse_state(self, state):
        #parse the state from the message. Columns are separated by ; and rows by ,
        state = state.split(";")
        res = []
        for i in range(len(state)):
            if(state[i] == "" or state[i] == ''):
                continue
            state[i] = state[i].split(",")
            for j in range(len(state[i])):
                if(state[i][j] == "" or state[i][j] == ''):
                    continue
                res.append(state[i][j])
            
        res= np.array(res, dtype=np.float32).reshape(1,27)
        return res

# # test parse_state
# state = "0,0,0,-1,0,0,-1,0,0;0,0,0,0,1,0,0,0,0;0,0,0,0,0,0,0,0,0;"
# env = Environment()
# print(env.parse_state(state))

    # def parse_state(self, state):
    #     #parse the state from the message. Columns are separated by ; and rows by ,
    #     state = state.split(";")
    #     state_widht = len(state[0].split(","))
    #     state_height = len(state)
    #     res = np.zeros((state_height, state_widht),dtype=np.float32)
    #     for i in range(len(state)):
    #         if(state[i] == "" or state[i] == ''):
    #             continue
    #         state[i] = state[i].split(",")
    #         for j in range(len(state[i])):
    #             if(state[i][j] == "" or state[i][j] == ''):
    #                 continue
    #             res[i][j] = state[i][j]
            
    #     return res

    def summary(self):
        reward, state, done = self.get_info()
        
        #transforme the reward, state and done to a json format with value, shape and type
        reward = {"value": reward, "shape": 1, "type": str(type(reward))}
        state = {"value": state, "shape": state.shape, "type": str(state.dtype)}
        done = {"value": done, "shape": 1, "type": str(type(done))}

        print("reward: ", reward)
        print("state: ", state)
        print("done: ", done)