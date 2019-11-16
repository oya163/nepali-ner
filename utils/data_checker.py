#!/usr/bin/env python3

# Simple program to check data statistics of NER file
# Input file should be in standard Stanford format
# Outputs number of PER, LOC, ORG tags

import csv
import argparse
import os
from collections import Counter

def main():
	parser = argparse.ArgumentParser(description='Input file name')
	
	parser.add_argument('filename', metavar='STR',
                    help='input valid file path')
					
	args = parser.parse_args()
	
	input_file = args.filename
	
	counter_test = 0
	per_list_test = []
	org_list_test = []
	loc_list_test = []
	with open(input_file,'r', encoding='utf-8') as in_file:
		reader = csv.reader(in_file, delimiter='\t')
		for row in reader:
			if len(row) == 0:
				counter_test += 1
			else:
				if row[1] == 'PER':
					per_list_test.append(row[0])
				if row[1] == 'ORG':
					org_list_test.append(row[0])
				if row[1] == 'LOC':
					loc_list_test.append(row[0])
					
				
	print("Total unique PERSON = ", len(per_list_test))
	# print("ORGANISATION distribution")
	# print(Counter(org_list))
	print("Total unique ORGANISATION = ", len(org_list_test))
	# print("LOCATION distribution")
	# print(Counter(loc_list))
	print("Total unique LOCATION = ", len(loc_list_test))
	print("Total number of sentences annotated = ", counter_test)
	
if __name__ == "__main__":
	main()