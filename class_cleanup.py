# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 00:00:12 2018

@author: Abhishek S
"""
import pandas as pd
import numpy as np
import nltk
import os
import gc
import re

gc.collect()


class cleanup:
    def __init__(self,chunk_rate,origin,destination=None):
        self.chunk_rate=chunk_rate
        self.origin=origin
        if destination is None:
            self.destination=origin
        else:
            self.destination=destination
        self.destination=destination
        self.files=list()
        self.filepath=dict()
        self.new_name=dict()
        
    def list_files(self,pattern=None,flags=None,verbose=False):
        f=[]
        for dirpath,dirname,file in os.walk(self.origin):
            f+=file
            self.filepath.update({ i:dirpath for i in file})
        if pattern is None:
            self.files=f
        else:
            if flags is None:
                patt=re.compile(pattern)
            else:
                patt=re.compile(pattern,flags)
            matches=map(patt.finditer,f)
            for match in matches:
                for fl in match:
                    self.files.append(fl.group(0))
        if verbose:
            print(self.files)

    def decorator(self,original_func):
        def wrapper_function(*args,**kwargs):
            mp=list(map(original_func,self.files))
            return mp
        return wrapper_function
    
    def copy(self,filename):
        with open(os.path.join(self.filepath[filename],filename),'rb') as rf:
            with open(os.path.join(self.destination,filename),'wb') as wf:
                line=rf.read(self.chunk_rate)
                while(len(line)>0):
                    wf.write(line)
                    line=rf.read(self.chunk_rate)
    def copy_files(self):
        cp=self.decorator(self.copy)
        cp()
    def rename_files(self):
        try:
            for f in self.files:
                os.rename(os.path.join(self.destination,f),os.path.join(self.destination,self.new_name[f]))
        except KeyError as e:
            print("New names have not been generated")
            print(e)
    def generate_new_names(self,pattern,replacement):
        patterns=list(pattern)
        replacement=list(replacement)
        assert(len(patterns)==len(replacement))
        new_name=self.files.copy()
        for patt,repl in zip(patterns,replacement): 
            pat=re.compile(patt)
            new_name=[pat.sub(repl,file) for file in new_name]
        self.new_name=dict(zip(self.files,new_name))
        return new_name
    
origin=r"G:\TV Serials\Naruto\Naruto Shippuden"
youtube=cleanup(1024*10,origin)
youtube.list_files(pattern=r"(Naruto|NS).*\.(mkv|avi|mp4)",flags=re.I)
youtube.files
youtube.filepath
new_name=youtube.generate_new_names([r"\(.{1,12}\)",r"_-_",r"(__| - Watch Naruto Shippuuden  Episode \d* online in high quality|^_|L@mBerT)",r"_\.",r"(NS|Naruto Shippu*den)",r"[^(\d{3}|\d{3}\-\d{3}|Naruto_Shippuuden|\.mkv|\.mp4|\.MP4 |\-)]"],["","_","",".","Naruto_Shippuuden",""])
new_name=youtube.generate_new_names([r"\(.{1,12}\)",r"_-_",r"(__| - Watch Naruto Shippuuden  Episode \d* online in high quality|^_|L@mBerT)",r"_\.",r"(NS|Naruto Shippu*den)"],["","_","",".","Naruto_Shippuuden"])



    
#files=youtube.list_files(r".*(mkv|avi|flv|MKV|mp4)",verbose=True)
#origin=r"E:\Python\test"
#destin=r"E:\Python\test\tcopy"
#os.chdir(origin)
#f=[]
#d=dict()
#for dirpath,dirname,files in os.walk(origin):
#    f+=files
#    print(dirpath)
#    print(files)
#    for i in files:
#        d[i]=dirpath
    
#
#
#mp=map(copy,f)
#list(mp)
#for _ in mp:
#    
#del destination
#
#def copy(file,destination):
#    with open(os.path.join(origin,file),'rb') as rf:
#        with open(os.path.join(destination,file),'wb') as wf:
#            line=rf.read(1024)
#            while(len(line)>0):
#                wf.write(line)
#                line=rf.read(1024)
#copy("titanic.csv")
#map(copy,)
#
