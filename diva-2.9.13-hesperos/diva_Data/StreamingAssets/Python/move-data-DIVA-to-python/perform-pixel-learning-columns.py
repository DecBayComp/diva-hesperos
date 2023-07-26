import time
import zmq
import numpy as np
import csv

if __name__ == '__main__':

    ### LOAD TAGS ###

    # parse pixel tags CSV to python dictionary
    tags_dictionary = {}
    tags_dictionary_np = {}
    with open("pixel_tags.csv", mode="r") as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        
        label_list = []
        for row in csv_reader:
            if (line_count == 0):
                for r in row:
                    tags_dictionary[r] = []
                    label_list.append(r)
            else:
                for r in range(len(row)):
                    tags_dictionary[label_list[r]].append(row[r])
            line_count += 1
        
        # convert to np array
        for row in tags_dictionary.keys():
            r = 0
            tags_dictionary_np[label_list[r]] = np.array(tags_dictionary[row])
            r += 1
            
        print("Pixel tags line count = " + str(line_count))
    
    ### PIXEL LEARNING ###


    
    ### LOAD PIXEL FEATURES ###

    # parse pixel features CSV to python dictionary
    pixel_features_dictionary = {}
    pixel_features_dictionary_np = {}
    with open("pixel_features.csv", mode="r") as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        
        label_list = []
        for row in csv_reader:
            if (line_count == 0):
                for r in row:
                    pixel_features_dictionary[r] = []
                    label_list.append(r)
            else:
                for r in range(len(row)):
                    pixel_features_dictionary[label_list[r]].append(row[r])
            line_count += 1
        
        # convert to np array
        for row in pixel_features_dictionary.keys():
            r = 0
            pixel_features_dictionary_np[label_list[r]] = np.array(tags_dictionary[row])
            r += 1
        
        print("Pixel features line count = " + str(line_count))
        
    ### PIXEL CLASSIFICATION/PREDICTION ###


    
    exit()

