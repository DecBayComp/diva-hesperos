import socket
import sys
import numpy as np
import pandas as pd
import warnings
import argparse

from train import combining_classifiers

# constants
DEFAULT_BUFFER_SIZE = 64

def Receive(connection, remove_padding, buffer_size = DEFAULT_BUFFER_SIZE):
    data = b''
    msg_len = 0
    while True:
        message = connection.recv(buffer_size - msg_len)
        data += message
        msg_len = len(data)
        if len(data) == buffer_size:
            if remove_padding:
                data = data.replace(b'\x00',b'')
            return data

def Send(connection, add_padding, buffer, buffer_size = DEFAULT_BUFFER_SIZE):
    if add_padding:
        connection.sendall(buffer + b'\x00' * (buffer_size - len(buffer)))
    else:
        print(str(buffer) + " Sent")
        connection.sendall(buffer)

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-name', type=str, help="server name", required=True)
    parser.add_argument('--server-port', type=int, help="server port", required=True)
    parser.add_argument('--classifier-path', type=str, help="classifier file (.pckl) path", required=True)
    parser.add_argument('--classification-strength', type=int, help="classification strength (value between 1 and 10)")
    args = parser.parse_args()

    # create TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind socket to the port
    server_address = (args.server_name, args.server_port)
    print("\nStarting up: " + str(server_address) +"\n")
    sock.connect(server_address)
    print("Connected...\n")

    connection_finished = False

    image_features = {}

    while True:
        # boolean to break from loop
        if connection_finished:
            break

        # receive size of tags array
        size_data = Receive(sock, True)
        SIZE = int(size_data)
        print("Size of brick: " + str(SIZE))

        # receive number of bricks/packets
        number_of_bricks_data = Receive(sock, True)
        NUMBER_OF_BRICKS = int(number_of_bricks_data)
        print("Number of bricks: " + str(NUMBER_OF_BRICKS))

        # receive number of features
        number_of_features_data = Receive(sock, True)
        NUMBER_OF_FEATURES = int(number_of_features_data)
        print("Number of features: " + str(NUMBER_OF_FEATURES) + "\n")

        # print classification classification_strength
        print("Classification Strength: " + str(5) + " / " + str(10))

        current_feature = 0
        current_brick =  0

        # condition reception of tags based on whether size of packet received
        if size_data:
            # receive/send operations with Unity
            while True:
                # receive label from Unity
                label_data = Receive(sock, True)
                if label_data:
                    # receive data according to feature array size
                    feature_data = Receive(sock, False, SIZE)

                    # add to image feature dictionary
                    image_features[label_data] = np.frombuffer(feature_data, dtype=np.uint8).astype(np.float)
                    # print("Received: {} {}".format(label_data, image_features[label_data]))

                    # increment feature received
                    current_feature += 1
                else:
                    connection_finished = True
                    break

                # increment brick if number of features attained
                if current_feature == NUMBER_OF_FEATURES:

                    print("Brick # " + str(current_brick))

                    # reset number of features
                    current_feature = 0
                    print("\nInferring... {0:.0f}%".format(100*(current_brick+1)/NUMBER_OF_BRICKS))
                    current_brick += 1

                    ###############################
                    # PERFORM CLASSIFICATION HERE #
                    ###############################

                    glob_class = combining_classifiers(pd.DataFrame({str(label): image_features[label] for label in image_features}), args.classifier_path, args.classification_strength)
                    glob_class.__prepare_data_for_inference__()

                    log_proba = glob_class.__predict_log_proba__()
                    proba = np.exp(log_proba)

                    # for i in range(100):
                    #     print(str(log_proba[i]),str(proba[i]))

                    proba = (1 - proba) * 255
                    proba_uint8 = proba.astype(np.uint8)

                    sock.sendall(proba_uint8.tobytes())

                    # if current_brick == 1:
                    #     N = len(log_proba)
                    #
                    #     file = open("proba.csv","w+")
                    #     for i in range(N):
                    #         file.write(str(proba[i]) + "\n")
                    #     file.close()

                        # log_proba = log_proba/255
                        #
                        # file = open("log_proba.csv","w+")
                        # for i in range(N):
                        #     file.write(str(log_proba[i]) + "\n")
                        # file.close()

                    ###############################
                    ###############################
                    ###############################

                    # exit program after all bricks classified
                    if current_brick == NUMBER_OF_BRICKS:
                        print("Inference finished; exiting program")
                        connection_finished = True
                        break
        else:
            break

    # clean up connection
    sock.close()

    # exit script
    exit()
