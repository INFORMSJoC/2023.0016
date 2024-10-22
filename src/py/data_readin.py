import numpy as np
import os

# Data Readin function
# fileName: users should provivde the relative file path for function to readin.
def data_readin(fileName):
    input_file = open(fileName, "r")
    raw_data_string = [elem for elem in input_file.readlines() if elem != '\n']
    input_file.close()
    
    raw_data_dict = {}
    raw_data_dist = []
    raw_data_dist_matrix = []
    
    test_data_dict = {}
    flip_keys = []
    flip_dists = []

    front_index = 1
    back_index = front_index+1
    switch_indicator = 0
    
    for index in range(len(raw_data_string)):
        raw_data_string[index] = raw_data_string[index].replace('\n', "")
        raw_data_string[index] = raw_data_string[index].split()
        if index == 0:
            num_targets = int(raw_data_string[index][0])
            num_views = int(raw_data_string[index][1])
        else:
            raw_data_dist=[]
            for item in raw_data_string[index]:
                raw_data_dist.append(int(item))
            raw_data_dist_matrix.append(raw_data_dist)
            switch_indicator = switch_indicator +1
            if switch_indicator > num_targets-1:
                raw_data_dict[(front_index, back_index)] = raw_data_dist_matrix 
                raw_data_dist_matrix = []
                back_index = back_index +1
                if back_index > num_views:
                    front_index = front_index +1
                    back_index = front_index +1
                switch_indicator = 0
                
    for key in raw_data_dict.keys():
        test_dists = raw_data_dict[key]
        test_dists_array = np.array([np.array(dist) for dist in test_dists])
        test_data_dict[key] = test_dists_array
        flip_key = (key[1], key[0])
        flip_dist_array = test_dists_array.T
        flip_dist = flip_dist_array.tolist()

        flip_keys.append(flip_key)
        flip_dists.append(flip_dist)

    for i in range(len(flip_keys)):
        flip_key = flip_keys[i]
        flip_dist = flip_dists[i]
        raw_data_dict[flip_key] = flip_dist
        
    return num_targets, num_views, raw_data_dict

def main():
    filePath = 'dist_data/'
    fileName = str(4) + 'D' + str(3) + '-' + str(1) + '.dat'
    completeFileName = os.path.join(filePath, fileName)
    num_targets, num_views, raw_data_dict = data_readin(completeFileName)
    print(num_targets)
    print(num_views)
    print(raw_data_dict)
        
if __name__ == "__main__":
    main()