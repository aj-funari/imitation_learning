import os

"""
Read through folder where images are being stored and write each label
with reformatted action to csv file. This step is needed for creating
customized dataset.
    - Example: 1.0-1.0-14:10:45-.jpeg, ['1.0', '1.0']
"""

def csv_file():
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

        action = [x,z]

        f.write(f"{save_label}, {action} \n")

if __name__ == "__main__":
    csv_file()
