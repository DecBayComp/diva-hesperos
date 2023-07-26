import socket
import sys
import numpy as np
import pandas as pd
from train import combining_classifiers
import pickle
import warnings
import argparse

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

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-name', type=str, help="server name", required=True)
    parser.add_argument('--server-port', type=int, help="server port", required=True)
    parser.add_argument('--classifier-path', type=str, help="classifier file (.pckl) path") # not required for training
    parser.add_argument('--classification-strength', type=int, help="classification strength (value between 1 and 10)")
    args = parser.parse_args()

    # print("classification strength = " + str(args.classification_strength))

    # create TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind socket to the port
    server_address = (args.server_name, args.server_port)
    print("\nStarting up: " + str(server_address) +"\n")
    sock.connect(server_address)
    print("Connected...\n")

    connection_finished = False

    tags = {}

    while True:

        # boolean to break from loop
        if connection_finished:
            break

        # receive size of tags array
        size_data = Receive(sock, True)
        SIZE = int(size_data)
        # print("Size of tags: " + str(SIZE))

        # receive number of features
        number_of_features_data = Receive(sock, True)
        NUMBER_OF_FEATURES = int(number_of_features_data)
        # print("Number of features (+ label): " + str(NUMBER_OF_FEATURES))
        # print("")

        print("Received inference-related metadata from DIVA\n")

        current_feature = 0

        # condition reception of tags based on whether size of tags received
        if size_data:
            # receive/send operations with Unity
            while True:
                # receive label from Unity
                label_data = Receive(sock, True)
                if label_data:
                    # receive data according to tags array size
                    tags_data = Receive(sock, False, SIZE)

                    # add to tags dictionary
                    tags[label_data] = np.frombuffer(tags_data, dtype=np.uint8).astype(np.float)
                    # print("Received: {} {}".format(label_data, tags[label_data]))

                    # increment feature received
                    current_feature += 1
                else:
                    connection_finished = True
                    break

                # increment brick if number of features attained
                if current_feature == NUMBER_OF_FEATURES:
                    print("Training data received; exiting program\n")
                    connection_finished = True
                    break
        else:
            break

    # clean up connection
    sock.close()

    ######################################
    # file = open("tags-exemple.csv","w+")

    # keys = list(tags.keys())
    # N = len(tags[keys[0]])
    #
    # for i in range(N):
    #     output_string = ""
    #
    #     for label in tags.keys():
    #         output_string += str(tags[label][i]) + ","
    #
    #     output_string = output_string[0:len(output_string)-1]
    #     output_string += "\n"
    #     file.write(output_string)
    #
    # file.close()
    ######################################

    # start training
    glob_class = combining_classifiers(pd.DataFrame({str(label): tags[label] for label in tags}), args.classifier_path, args.classification_strength)
    glob_class.__prepare_data_for_training__()
    #glob_class.__recreate_all_features_from_previous_learners__()
    glob_class.__apply_multiple_classifier_to_features__()
    glob_class.__apply_main_classifier__()

    # print(glob_class.liste_learner)

    if args.classifier_path != "":
        # filename = "liste_classifier.pckl"
        pickle.dump(glob_class.liste_learner, open(args.classifier_path, 'wb'))

    # exit script
    exit()
