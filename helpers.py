import os
import cv2

"""
Read through folder where images are being stored and write each label
with reformatted action to csv file. This step is needed for creating
customized dataset.
    - Example: 1.0-1.0-14:10:45-.jpeg, ['1.0', '1.0']
"""

def csv_save_to_file():
    f = open("/home/aj/images/labels/avoid_walls_labels.csv", 'w')
    DATADIR = '/home/aj/images/avoid_walls/'
    
    for label in os.listdir(DATADIR):
        save_label = label
        label = label.split('-')

        if len(label) == 4:  # positive x and z coordinates
            x = label[0]
            z = label[1]
            action = [x,z]
        
        if len(label) == 5:  # negative x or z coordinate
            if label[0] == '': # -x
                x = '-' + label[1]
                z = label[2]
            if label[1] == '': # -z
                x = label[0]
                z = '-' + label[2]

        if len(label) == 6:  # negative x and z coordinates
            x = '-' + label[1]
            z = '-' + label[3]

        action = (x,z)

        f.write(f"{save_label}, {action} \n")  # save image label and action to csv file


"""
Reformat linear/angular velocities in csv file from string
into floats. Float value is needed for MSE function. 
"""

def csv_format(x, z):
    
    x = x.replace("(", "")
    x = x.strip()  # remove white space
    x = x.strip("'")

    z = z.replace(")", "")
    z = z.strip()
    z = z.strip("'")

    return(float(x), float(z))

"""
Resize image to match input with neural network
- function is being called in data_collector.py
- image is resized before being saved in folder
"""

def resize(img):  # input: 3 x 768 x 1024
    dim = (224, 224)  # output: 224 x 224 x 3
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return(img)

if __name__ == "__main__":
    csv_save_to_file()
