import time
import zmq
import numpy as np


class Socket:
    def __init__(self, addr="tcp://*:5555"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(addr)

    def Receive(self):
        label = self.socket.recv()
        
        if label == b"Next":
            return 1

        if label == b"Finish":
            print("Last brick")
            return

        self.socket.send(b"Label Echo")

        data = self.socket.recv()

        data_formatted = np.frombuffer(data, dtype=np.uint8)
        self.socket.send(b"Data Echo")

        return label, data_formatted

    def ReceiveAll(self):
        bricks = []
        dictionary = {}
        
        brick_number = 1
        dummy_array = bytearray(262144)

        # loop to receive all features
        while True:

            rep = socket.Receive()

            if rep is None: # send received data back (testing)
                print("Brick " + str(brick_number) + " received")
                bricks.append(dictionary)
                dictionary = {}
                
                # send data for classification
                self.socket.send(dummy_array)
                brick_number = brick_number + 1
                break
                
            elif rep == 1: # next brick
                print("Brick " + str(brick_number) + " received")
                bricks.append(dictionary)
                dictionary = {}
                
                # send data for classification
                self.socket.send(dummy_array)
                brick_number = brick_number + 1
                
            else: # save to features dictionary
                label, data_formatted = rep
                dictionary[label] = data_formatted

        return bricks


if __name__ == '__main__':
    socket = Socket()
    bricks = socket.ReceiveAll()

    file = open("data.csv","w+")

    for dict in bricks:
        keys = list(dict.keys())
        N = len(dict[keys[0]])
        
        for i in range(N):
            output_string = ""
            
            for label in dict.keys():
                output_string += str(dict[label][i]) + ","
                
            output_string = output_string[0:len(output_string)-1]
            output_string += "\n"
            file.write(output_string)

    file.close()

    exit()

