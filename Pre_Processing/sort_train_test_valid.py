import json
from glob import glob
import os
from argparse import ArgumentParser
import random
import numpy as np
from shutil import copyfile

if __name__ == '__main__':
   
    descr = """
    Each images should have its own sub-folder with the svs image name followed by '_files'
    Each images should have subfolders with names corresponding to the magnification associated with the jpeg tiles saved inside it
    The sorting will be done using tiles corresponding to a magnification of 20 (+/- 5 if the 20 folder does not exist)
    15%% will be put for validation, 15%% for testing and the leftover for training. However, if split is > 0, then the data will be split in train/test only in "# split" non-overlapping ways (each way will have 100/(#split) % of test images).
    linked images' names will start with 'train_', 'test_' or 'valid_' followed by the svs name and the tile ID
    
    """
    ## Define Arguments
    parser = ArgumentParser(description=descr)
    parser.add_argument("--SourceFolder", help="path to tiled images", dest='SourceFolder')
    parser.add_argument("--JsonFile", help="path to metadata json file", dest='JsonFile') #Used in the case of selecting a particular tumor type. This file can be downlaoded along with manifest file
    parser.add_argument("--Magnification", help="magnification to use", type=float, dest='Magnification')
    parser.add_argument("--MagDiffAllowed", help="difference allowed on Magnification", type=float,
                        dest='MagDiffAllowed')
    parser.add_argument("--PercentValid", help="percentage of images for validation (between 0 and 100)", type=float,dest='PercentValid')
    parser.add_argument("--PercentTest", help="percentage of images for testing (between 0 and 100)", type=float,dest='PercentTest')
    parser.add_argument("--PatientID",help="Patient ID is supposed to be the first PatientID characters (integer expected) of the folder in which the pyramidal jpgs are. Slides from same patient will be in same train/test/valid set. This option is ignored if set to 0 or -1 ",type=int, dest='PatientID') #Patient Id for this data set has 12 characters



    ## Parse Arguments
    args = parser.parse_args()

    if args.JsonFile is None:
        print("No JsonFile found")
        args.JsonFile = ''

    if args.PatientID is None:
        print("PatientID ignored")
        args.PatientID = 0

    SourceFolder = os.path.abspath(args.SourceFolder)
    imgFolders = glob(os.path.join(SourceFolder, "*_files"))
    random.shuffle(imgFolders)  # randomize order of images

    JsonFile = args.JsonFile
    if '.json' in JsonFile:
        with open(JsonFile) as fid:
            jdata = json.loads(fid.read())
        try:
            jdata = dict((jd['file_name'].replace('.svs', ''), jd) for jd in jdata)
        except:
            jdata = dict((jd['Patient ID'], jd) for jd in jdata)

    Magnification = args.Magnification
    MagDiffAllowed = args.MagDiffAllowed
    sort_function = "All"

    PercentValid = args.PercentValid / 100.
    if not 0 <= PercentValid <= 1:
        raise ValueError("PercentValid is not between 0 and 100")
    PercentTest = args.PercentTest / 100.
    if not 0 <= PercentTest <= 1:
        raise ValueError("PercentTest is not between 0 and 100")
    ## Main Loop
    print("******************")
    Classes = {}
    NbrTilesCateg = {}
    PercentTilesCateg = {}
    NbrImagesCateg = {}
    PercentSlidesCateg = {}
    Patient_set = {}
    NbSlides = 0
    ttv_split = {}
    nbr_valid = {}

    #print("imgFolders")
    #print(imgFolders)
    for cFolderName in imgFolders:

        NbSlides += 1
        imgRootName = os.path.basename(cFolderName)
        imgRootName = imgRootName.replace('_files', '')
     
        try:
            image_meta = jdata[imgRootName]
        except KeyError:
            try:
                image_meta = jdata[imgRootName[:args.PatientID]]
            except KeyError:
                print("file_name %s not found in metadata" % imgRootName[:args.PatientID])
                continue
        SubDir = sort_function
        print("SubDir is %s" % SubDir)
        
        SetDir = ""
        SubDir=os.path.join(SourceFolder,SubDir)
	print(SubDir)
	if not os.path.exists(SubDir):
            os.makedirs(SubDir)
        try:
            Classes[SubDir].append(imgRootName)
        except KeyError:
            Classes[SubDir] = [imgRootName]

        # Check in the reference directories if there is a set of tiles at the desired magnification
        AvailMagsDir = [x for x in os.listdir(cFolderName)
                        if os.path.isdir(os.path.join(cFolderName, x))]
        AvailMags = tuple(float(x) for x in AvailMagsDir)
        # check if the mag was known for that slide
        if max(AvailMags) < 0:
            print("Magnification was not known for that file.")
            continue
        mismatch, imin = min((abs(x - Magnification), i) for i, x in enumerate(AvailMags))
        if mismatch <= MagDiffAllowed:
            AvailMagsDir = AvailMagsDir[imin]
        else:
            # No Tiles at the mag within the allowed range
            print("No Tiles found at the mag within the allowed range.")
            continue

        # Copy/symbolic link the images into the appropriate folder-type
        SourceImageDir = os.path.join(cFolderName, AvailMagsDir, "*") # we can also symlink the files.
        AllTiles = glob(SourceImageDir)

        if SubDir in NbrTilesCateg.keys():
            print("%s Already in dictionary" % SubDir)
        else:
            NbrTilesCateg[SubDir] = 0
            NbrTilesCateg[SubDir + "_train"] = 0
            NbrTilesCateg[SubDir + "_test"] = 0
            NbrTilesCateg[SubDir + "_valid"] = 0
            PercentTilesCateg[SubDir + "_train"] = 0
            PercentTilesCateg[SubDir + "_test"] = 0
            PercentTilesCateg[SubDir + "_valid"] = 0
            NbrImagesCateg[SubDir] = 0
            NbrImagesCateg[SubDir + "_train"] = 0
            NbrImagesCateg[SubDir + "_test"] = 0
            NbrImagesCateg[SubDir + "_valid"] = 0
            PercentSlidesCateg[SubDir + "_train"] = 0
            PercentSlidesCateg[SubDir + "_test"] = 0
            PercentSlidesCateg[SubDir + "_valid"] = 0

        NbTiles = 0
        ttv = 'None'
        if len(AllTiles) == 0:
            continue
        for TilePath in AllTiles:
            TileName = os.path.basename(TilePath)
            NbTiles += 1
            # rename the images with the root name, and put them in train/test/valid
            if (PercentSlidesCateg.get(SubDir + "_test") <= PercentTest) and (PercentTest > 0):
                ttv = "test"
            elif (PercentSlidesCateg.get(SubDir + "_valid") <= PercentValid) and (PercentValid > 0):
                ttv = "valid"
            else:
                ttv = "train"
            # If that patient had an another slide/scan already sorted, assign the same set to this set of images
            if args.PatientID > 0:
                Patient = imgRootName[:args.PatientID]
                if Patient in Patient_set:
                    ttv = Patient_set[Patient]
                else:
                    Patient_set[Patient] = ttv

            NewImageDir = os.path.join(SubDir, "_".join((ttv, imgRootName, TileName)))  # all train initially
            if not os.path.lexists(NewImageDir):
                os.rename(TilePath, NewImageDir)
        # update stats

        if ttv == "train":
            NbrTilesCateg[SubDir + "_train"] = NbrTilesCateg.get(SubDir + "_train") + NbTiles
            NbrImagesCateg[SubDir + "_train"] = NbrImagesCateg[SubDir + "_train"] + 1
        elif ttv == "test":
            NbrTilesCateg[SubDir + "_test"] = NbrTilesCateg.get(SubDir + "_test") + NbTiles
            NbrImagesCateg[SubDir + "_test"] = NbrImagesCateg[SubDir + "_test"] + 1
        elif ttv == "valid":
            NbrTilesCateg[SubDir + "_valid"] = NbrTilesCateg.get(SubDir + "_valid") + NbTiles
            NbrImagesCateg[SubDir + "_valid"] = NbrImagesCateg[SubDir + "_valid"] + 1
        else:
            continue
        NbrTilesCateg[SubDir] = NbrTilesCateg.get(SubDir) + NbTiles
        NbrImagesCateg[SubDir] = NbrImagesCateg.get(SubDir) + 1

        PercentTilesCateg[SubDir + "_train"] = float(NbrTilesCateg.get(SubDir + "_train")) / float(
            NbrTilesCateg.get(SubDir))
        PercentTilesCateg[SubDir + "_test"] = float(NbrTilesCateg.get(SubDir + "_test")) / float(
            NbrTilesCateg.get(SubDir))
        PercentTilesCateg[SubDir + "_valid"] = float(NbrTilesCateg.get(SubDir + "_valid")) / float(
            NbrTilesCateg.get(SubDir))
        PercentSlidesCateg[SubDir + "_train"] = float(NbrImagesCateg.get(SubDir + "_train")) / float(
            NbrImagesCateg.get(SubDir))
        PercentSlidesCateg[SubDir + "_test"] = float(NbrImagesCateg.get(SubDir + "_test")) / float(
            NbrImagesCateg.get(SubDir))
        PercentSlidesCateg[SubDir + "_valid"] = float(NbrImagesCateg.get(SubDir + "_valid")) / float(
            NbrImagesCateg.get(SubDir))



