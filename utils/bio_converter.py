#!/usr/bin/python

import os
import sys

def main():
    label_list = ['PER', 'ORG', 'LOC']
    with open(sys.argv[1], 'r', encoding='utf-8') as in_file, open(sys.argv[2], 'w', encoding='utf-8') as out_file:
        prev_label = ' '
        for i1, row in enumerate(in_file):
            row = row.strip().split()
            if len(row) > 1:
                label = row[-1]
                if prev_label[0] != 'O' and prev_label[2:] == label:
                    label = 'I-' + label
                elif label in label_list:
                    label = 'B-' + label
                #out_file.write(row[0]+'\t'+row[1]+'\t'+label+'\n')
                out_file.write(row[0]+'\t'+label+'\n')
                prev_label = label
            else:
                out_file.write('\n')


if __name__ == "__main__":
    main()
