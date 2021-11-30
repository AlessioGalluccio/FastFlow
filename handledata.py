from pathlib import Path
from distutils.dir_util import copy_tree
import config as c
import os
import shutil

def handledata():
    Path("./dataset/"+ c.class_name).mkdir(parents=True, exist_ok=True)
    Path("./dataset/"+ c.class_name + "/train").mkdir(parents=True, exist_ok=True)
    Path("./dataset/"+ c.class_name + "/test").mkdir(parents=True, exist_ok=True)
    Path("./dataset/"+ c.class_name + "/train/good").mkdir(parents=True, exist_ok=True)
    Path("./dataset/"+ c.class_name + "/test/anomaly").mkdir(parents=True, exist_ok=True)
    Path("./dataset/"+ c.class_name + "/test/good").mkdir(parents=True, exist_ok=True)

    copy_directory("./data/mvtec/"+ c.class_name + "/train/good", "./dataset/"+ c.class_name + "/train/good")
    copy_directory("./data/mvtec/"+ c.class_name + "/test/good", "./dataset/"+ c.class_name + "/test/good")

    # TOOTHBRUSH
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/defective", "./dataset/"+ c.class_name + "/test/anomaly")

    # CAPSULE
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/crack", "./dataset/"+ c.class_name + "/test/anomaly")
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/faulty_imprint", "./dataset/"+ c.class_name + "/test/anomaly")
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/poke", "./dataset/"+ c.class_name + "/test/anomaly")
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/scratch", "./dataset/"+ c.class_name + "/test/anomaly")
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/squeeze", "./dataset/"+ c.class_name + "/test/anomaly")

    # GRID
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/bent", "./dataset/"+ c.class_name + "/test/anomaly")
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/broken", "./dataset/"+ c.class_name + "/test/anomaly")
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/glue", "./dataset/"+ c.class_name + "/test/anomaly")
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/metal_contamination", "./dataset/"+ c.class_name + "/test/anomaly")
    #copy_directory("./data/mvtec/"+ c.class_name + "/test/thread", "./dataset/"+ c.class_name + "/test/anomaly")

    # HAZELNUT
    copy_directory("./data/mvtec/"+ c.class_name + "/test/crack", "./dataset/"+ c.class_name + "/test/anomaly", "crack")
    copy_directory("./data/mvtec/"+ c.class_name + "/test/cut", "./dataset/"+ c.class_name + "/test/anomaly", "cut")
    copy_directory("./data/mvtec/"+ c.class_name + "/test/hole", "./dataset/"+ c.class_name + "/test/anomaly", "hole")
    copy_directory("./data/mvtec/"+ c.class_name + "/test/print", "./dataset/"+ c.class_name + "/test/anomaly", "print")



def copy_directory(fromDirectory, toDirectory, tag = None):
    if tag == None:
        copy_tree(fromDirectory, toDirectory)
    else:
        for root, dirs, files in os.walk(fromDirectory):
            for filename in files:
                # I use absolute path, case you want to move several dirs.
                source = os.path.join( os.path.abspath(root), filename )

                # Separate base from extension
                base, extension = os.path.splitext(filename)

                # Initial new name
                new_name = os.path.join(toDirectory, base + "_" + tag + extension)

                shutil.copy(source, new_name)