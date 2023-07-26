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
        print(str(label))
        
        if label == b"Finish":
            return

        self.socket.send(b"Label Echo")

        data = self.socket.recv()

        data_formatted = np.frombuffer(data, dtype=np.uint8)
        self.socket.send(b"Data Echo")

        return label, data_formatted

    def ReceiveAll(self):
        tags = {}

        while True:
            # loop to receive all features

            rep = socket.Receive()
            if rep is None:
                break
            else:
                label, data_formatted = rep
                tags[label] = data_formatted
                # print(str(label) + " received")

        return tags


if __name__ == '__main__':
    socket = Socket()
    tags = socket.ReceiveAll()

    # for label in tags:
        # print("Received request: {} {}".format(label, tags[label]))

    file = open("tags.csv","w+")
    
    keys = list(tags.keys())
    N = len(tags[keys[0]])
    
    for i in range(N):
        output_string = ""
        
        for label in tags.keys():
            output_string += str(tags[label][i]) + ","
        
        output_string = output_string[0:len(output_string)-1]
        output_string += "\n"
        file.write(output_string)

    file.close()
    
    exit()

