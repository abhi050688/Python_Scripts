# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 22:51:46 2018

@author: Abhishek S
"""

import pandas as pd
import numpy as np
import nltk
import os
import gc
import re

gc.collect()

os.chdir('E:/')
#print(os.listdir())
origin='E:/Video/Boruto'
destination=os.path.join(origin,'Boruto_New')
os.mkdir(destination)

def copy_files(origin,filename,destination,wfilename):
    with open(os.path.join(origin,filename),'rb') as rf:
            with open(os.path.join(destination,wfilename),'wb') as wf:
                chunk_size=1024*4
                fread=rf.read(chunk_size)
                while len(fread)>0:
                    wf.write(fread)
                    fread=rf.read(chunk_size)

f,ext=os.path.splitext('Boruto- Naruto Next Generations Episode 066 - Watch Boruto- Naruto Next Generations Episode 066 online in high quality.mp4')
xr=r"\d{1,3}"
pattern=re.compile(xr)
matches=pattern.findall(f)
match=matches[0]
match
for match in matches:
    print(match)

f=sorted(f)
for  dirpath,dirname,filename in os.walk(origin):
    print("{}-{}-{}".format(dirpath,dirname,filename))

patt=re.compile(r".*(mkv|avi|flv|MKV|mp4)")
matches=patt.finditer('(15) Enrique Iglesias - Escape - YouTube.MKV')
#matches=map(patt.findall,filename)
for match in matches:
    print(match)






pattern=re.compile(r"\d{1,3}")
for f in os.listdir():
    filename,ext=os.path.splitext(f)
    match=pattern.findall(filename)[0]
    wfilename="Boruto_"+match+ext
    copy_files(origin,f,destination,wfilename)

for match in matches:
    for fl in match:
        print(fl)



def list_files(pattern,flags=None,verbose=False):
    f=[]
    files=[]
    for dirpath,dirname,file in os.walk(r"E:\YouTube Video"):
        f+=file
    else:
        if flags is None:
            patt=re.compile(pattern)
        else:
            patt=re.compile(pattern,re.I)
        matches=map(patt.finditer,f)
        for match in matches:
            for fl in match:
                files.append(fl.group(0))
    return files
files=list_files()

pattern=r".*(mkv|avi|flv|MKV|mp4)"
for dirpath,dirname,file in os.walk(r"E:\Python\test"):
    print(dirpath)
    print(dirname)
    print(file)




