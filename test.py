import pandas as pd

annotations_file = "/home/aj/images/labels/avoid_walls_labels.csv"
img_labels = pd.read_csv(annotations_file)
# print(img_labels)

idx = 0
x = img_labels.iloc[idx, 1] 
z = img_labels.iloc[idx, 2] 

"""
Reformat linear/angular velocities in csv file from string
into floats. Float value is needed for MSE function. 
"""

def format(x, z):
    
    x = x.replace("(", "")
    x = x.strip()  # remove white space
    x = x.strip("'")

    z = z.replace(")", "")
    z = z.strip()
    z = z.strip("'")

    return(float(x), float(z))

print(format(x, z))
