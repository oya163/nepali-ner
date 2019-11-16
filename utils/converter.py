#!/usr/bin/env python3
'''
    File converter into CoNLL format
    Author: Oyesh Mann Singh
    Date: 10/14/2019
'''

import os
import io
import argparse
import re

parser = argparse.ArgumentParser("POS Tagger Argument Parser")
parser.add_argument("-i", "--input_folder", default="../data/", metavar="PATH", help="Data folder path")
parser.add_argument("-f", "--input_file", default="../data/sample.txt", metavar="PATH", help="Input file path")
parser.add_argument("-o", "--output_file", default="sample_out.txt", metavar="PATH", help="Output file path")

args = parser.parse_args()

def main():
    path = args.input_folder
    out_file = os.path.join(path, args.output_file)

    with open(out_file, 'w', encoding='utf-8') as out_f:
        for input_file in os.listdir(path):
            curr_file = os.path.join(path, input_file)
            if os.path.isfile(curr_file):
                with open(curr_file, 'r', encoding='utf-8-sig') as in_f:
                    reader = in_f.readlines()
                    for row in reader:
                        # Split and extract words inside <> brackets
                        words = re.split(r"\<(.*?)\>", row)

                        # Separate token and tag
                        for i in range(0, len(words)-1, 2):
                            out_f.write(words[i].strip()+'\t'+words[i+1].strip()+'\n')
                        out_f.write('\n')


if __name__ == "__main__":
    main()

