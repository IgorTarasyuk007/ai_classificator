from PIL import Image
import os, sys
import pathlib
path = os.path.dirname(os.path.realpath(__file__))+'/tf_files/'
dir = pathlib.Path("tf_files/")

def resize_aspect_fit():
    for dirs in [i for i in os.walk('./tf_files/')][0][1]:
        print(dirs)
        i = 0
        for item in [i for i in os.walk(f'./tf_files/{dirs}')][0][1:]:
            for ii in item:
                os.rename(path+dirs+'/'+ii, path+dirs+'/'+dirs+"_"+str(i)+'.jpg')
                i += 1

resize_aspect_fit()