import os
import cv2
import torch
import pandas as pd

"""
Read through folder where images are being stored and write each label
with reformatted action to csv file. This step is needed for creating
customized dataset.
    - Example: 1.0-1.0-14:10:45-.jpeg, 1.0, 1.0
"""

def save_to_csv_file(DATADIR, labels_file):
    f = open(labels_file, 'w')  # open file

    """
    Loop through folder containing images to retrieve labels
    - Images are located in a different folder labels csv file. This
      is the read for multiple directories
    e.g. DATADIR, LABELDIR 
    """

    for label in os.listdir(DATADIR):
        save_label = label  # e.g. 1.5189604759216309-0.630268931388855-23:18:20-.jpeg
        # print(save_label)
        label = label.split('-')

        if len(label) == 4:  # positive x and z coordinates
            x = label[0]
            z = label[1]
        
        if len(label) == 5:  # negative x or z coordinate
            if label[0] == '':  # -x
                x = '-' + label[1]
                z = label[2]
            if label[1] == '':  # -z
                x = label[0]
                z = '-' + label[2]

        if len(label) == 6:    # negative x and z coordinates
            x = '-' + label[1]
            z = '-' + label[3]

        f.write(f"{save_label}, {x}, {z} \n")  # save to csv file
    f.close()  # close file

"""
Reformat linear/angular velocities in csv file from string
into floats.
- float or tensor value is needed for MSE function. 
"""

def csv_label_to_tensor(x, z):
    output = torch.zeros(2)  # tensor([0., 0.])
    output[0] = float(x)  # sets first index to float(x)
    output[1] = float(z)  # sets second index to float(z)
    return output

"""
Resize image to match input with neural network
- function is being called in data_collector.py
- image is resized before being saved in folder
"""

def resize(img):  # input: 3 x 768 x 1024
    dim = (224, 224)  # output: 224 x 224 x 3
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return(img)

# if __name__ == "__main__":
#     # csv_save_to_file()

#     annotations_file = "/home/aj/images/labels/avoid_walls_labels.csv"
#     img_labels = pd.read_csv(annotations_file)
#     # print(img_labels)

#     idx = 0
#     x = img_labels.iloc[idx, 1] 
#     z = img_labels.iloc[idx, 2]

#     print(format_action_for_csv_file(x, z))