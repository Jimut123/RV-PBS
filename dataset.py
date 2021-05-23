import argparse
import numpy as np
import glob
import os
from xml.dom import minidom
import shutil

np.random.seed(42069)

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int, default=80, help="Percentage of training dataset")
parser.add_argument("--test", type=int, default=10, help="Percentage of test dataset")
parser.add_argument("--val", type=int, default=10, help="Percentage of validation dataset")
parser.add_argument("-m", "--move", action='store_true', default=False, help="Move files to dataset instead of copying")
parser.add_argument("input_dir")
parser.add_argument("output_dir")

args = parser.parse_args()

if not (args.train + args.test + args.val) == 100:
    raise ValueError("Train, Test and Validation percentage should add up to 100")

class_names = os.listdir(args.input_dir)
os.mkdir(args.output_dir)

for class_name in class_names:
    dataset_class_dir = os.path.join(args.input_dir, class_name)
    dataset_class_dir_output = os.path.join(args.output_dir, class_name)

    if not os.path.isdir(dataset_class_dir):
        continue
    dataset_class_dir = os.path.join(args.input_dir, class_name)
    dataset_class_dir_output = os.path.join(args.output_dir, class_name)
    
    os.mkdir(dataset_class_dir_output)
    annotations = {}
    doc = minidom.parse(os.path.join(dataset_class_dir, "annotations.xml"))
    images = doc.getElementsByTagName('image')

    for image in images:
        image_name = image.attributes['name'].value
        annotations[image_name] = image
    
    image_names = list(annotations.keys())
    np.random.shuffle(image_names)
    
    train_no = int(round((args.train / 100) * len(image_names)))
    assert(train_no > 0)

    valid_no = int(round((args.val / 100) * len(image_names)))
    assert(valid_no > 0)

    test_no = len(image_names) - (train_no + valid_no)
    assert(test_no > 0)

    train_files = image_names[:train_no]
    valid_files = image_names[train_no:train_no + valid_no]
    test_files = image_names[train_no + valid_no:]


    for subset, arr in [ ("train", train_files), ("test", test_files), ("val", valid_files) ]:
        os.mkdir(os.path.join(dataset_class_dir_output, subset))
        annotation_xml = minidom.Document()
        annotations_root = annotation_xml.createElement('annotations')
        annotation_xml.appendChild(annotations_root)

        for f in arr:
            annotations_root.appendChild(annotations[f])
            if not args.move:
                shutil.copy2(os.path.join(dataset_class_dir, f), os.path.join(dataset_class_dir_output, subset, f))
            else:
                shutil.move(os.path.join(dataset_class_dir, f), os.path.join(dataset_class_dir_output, subset, f))
        xml_str = annotation_xml.toprettyxml(indent="\t")

        with open(os.path.join(dataset_class_dir_output, subset, "annotations.xml"), 'w') as f:
            f.write(xml_str)