#!/usr/bin/env python3
'''
    Converts 3 columns into 2 columns
    Author: Oyesh Mann Singh
    Date: 10/16/2019
'''

import os
import argparse
import pandas as pd
import numpy as np
import csv
import shutil

def preprocess(input_file, output_file):
    with open(input_file,'r', encoding='utf-8') as in_file, open(output_file,'w', encoding='utf-8') as out_file:
        for i, row in enumerate(in_file):
            #To know which line is defunct in file
            #print(i+1)
            row = row.strip().split("\t")
            if len(row) > 2:
                if row[0] != "-DOCSTART-":
                    out_file.write(row[0]+" "+row[-1]+"\n")
            else:
                out_file.write("\n")

        
def main(**args):
    input_file = args["input_file"]
    output_file = args["output_file"]
    
    preprocess(input_file, output_file)
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser("Dataset Splitter Argument Parser")
    parser.add_argument("-i", "--input_file", default="./data/ebiquity/stemmed/total.bio", metavar="PATH", help="Input file path")
    parser.add_argument("-o", "--output_file", default="./data/ebiquity/stemmed_2/total.bio", metavar="PATH", help="Output file path")

    args = vars(parser.parse_args())

    main(**args)
