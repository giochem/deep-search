import gdown
import zipfile
import os
import csv
import shutil

def covert_coco_to_pascal_voc(coco):
    x_min, y_min, width, height = coco
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]

gdown.download("https://drive.google.com/uc?id=134QOvaatwKdy0iIeNqA_p-xkAhkV4F8Y", "CrowdHuman_train01.zip", quiet=False)
gdown.download("https://drive.google.com/uc?id=17evzPh7gc1JBNvnW1ENXLy5Kr4Q_Nnla", "CrowdHuman_train02.zip", quiet=False)
gdown.download("https://drive.google.com/uc?id=1tdp0UCgxrqy1B6p8LkR-Iy0aIJ8l4fJW", "CrowdHuman_train03.zip", quiet=False)
gdown.download("https://drive.google.com/uc?id=18jFI789CoHTppQ7vmRSFEdnGaSQZ4YzO", "CrowdHuman_val.zip", quiet=False)

with zipfile.ZipFile("CrowdHuman_train01.zip", 'r') as zip_ref:
    zip_ref.extractall()
with zipfile.ZipFile("CrowdHuman_train02.zip", 'r') as zip_ref:
    zip_ref.extractall()
with zipfile.ZipFile("CrowdHuman_train03.zip", 'r') as zip_ref:
    zip_ref.extractall()
with zipfile.ZipFile("CrowdHuman_val.zip", 'r') as zip_ref:
    zip_ref.extractall()

os.mkdir("./data")
os.mkdir("./data/images")
os.mkdir("./data/labels")
for filename in os.listdir('./Images'):
    shutil.move(f"./Images/{filename}", "./data/images")

os.rmdir("Images")

with open("info.txt", "w") as f_w:
    f_w.write("""hbox fbox vbox fbox gender age\nformat: x_min, y_min, x_max, y_max\n""")
    f_w.write("""fbox: human full-body bounding-box\nvbox: human visible-region bounding-box\nhbox: head bounding-box\n""")

for filename in ["annotation_val_with_classes.csv", "annotation_train_with_classes.csv"]:
    with open(filename) as f_r:
        csv_reader = csv.reader(f_r, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            id, box_id, hbox, fbox, vbox, gender, age = row
            hbox, fbox, vbox = eval(hbox), eval(fbox), eval(vbox)
            hbox = covert_coco_to_pascal_voc(hbox)
            fbox = covert_coco_to_pascal_voc(fbox)
            vbox = covert_coco_to_pascal_voc(vbox)
            result = hbox + fbox + vbox + [gender] + [age]

            result = ', '.join( [str(val) for val in result])
            with open(f"./data/labels/{id}.txt", "a") as f_w:
                f_w.write(f"{result}\n")
            print(f"{idx}\r")