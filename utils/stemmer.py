#!/usr/bin/env python3
'''
    Stems the postpositions in
    given dataset brute-force approach
    
    Author: Oyesh Mann Singh
    Date: 12/08/2019
'''

import os
import argparse
import pandas as pd
import numpy as np
import csv
import shutil


def stem(pp_file, input_file, output_file):
    stemmers = open(pp_file, 'r', encoding='utf-8').readlines()[0].split()
    lemma_tag = 'O'
    
    in_file = open(input_file,'r',encoding='utf-8').readlines()
    out_file = open(output_file, 'w', encoding='utf-8')
    
    not_to_be_lemmatized=['एमाले', 'अमेरिका', 'अधिकारी', 'शङ्का', 'मात्रिका']

    for line in in_file:
        words=line.strip().split('\t')
        lemmatize=False
        if len(words) > 1:
            saved_lemma = ''
            if words[0] not in not_to_be_lemmatized:
                for pp in stemmers:
                    if words[0] == pp:
                        break
                    elif words[0].endswith(pp):
                        words[0]=words[0][:-len(pp)]
                        saved_lemma=pp
                        lemmatize=True
                        break
            if len(words[0]) > 0:
                out_file.write(words[0]+'\t'+words[1]+'\n')
                if lemmatize:
                    out_file.write(saved_lemma+'\t'+lemma_tag+'\n')
                    lemmatize=False
        else:
            out_file.write('\n')
            
    out_file.close()
    print("File lemmatized!!!")

        
def main(args):
    input_file = args.input_file
    output_file = args.output_file
    pp_file = args.pp_file
    
    stem(pp_file, input_file, output_file)
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser("Dataset Stemmer Argument Parser")
    parser.add_argument("-p", "--pp_file", default="./data/stemming/postpositions.txt", metavar="PATH", help="Postpositions file path")
    parser.add_argument("-i", "--input_file", default="./data/ebiquity_v2/total.bio", metavar="PATH", help="Input file path")
    parser.add_argument("-o", "--output_file", default="./data/ebiquity_v2/total_stem.bio", metavar="PATH", help="Output file path")

    args = parser.parse_args()

    main(args)
