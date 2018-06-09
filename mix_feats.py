#!/usr/bin/env python
'''
this script merges background and signal files (specified on cmd line)
into a single mixed features file
'''

import sys
import os

dir_str = sys.argv[1]
match_str = sys.argv[2]
files = []
infil = []

for f in os.listdir(dir_str):
    if match_str in f:
        files.append(dir_str + "/" + f)
        infil.append(open(dir_str + "/" + f, 'r'))

of = open('./training_files/' + match_str + '.ascii', 'w')

lin = []
for f in infil:
    lin.append(f.readline())

header = lin[0]
print(header)
of.write(header)

while len(lin[0]) > 2:
    lin = []
    for f in infil:
        lin.append(f.readline())
        #print(len(lin[-1].split(',')))
        of.write(lin[-1])

f.close()

#
# for i in range(len(files)):
#     infil.append(open(files[i],'r')
