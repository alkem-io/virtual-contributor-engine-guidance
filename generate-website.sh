#!/bin/bash
# arguments:
# 1: git repo
# 2: website destination directory
# 3: website source directory
echo About to generate website
rm -r -f $3
git clone $1 $3
cd $3
export PATH=$PATH:/usr/local/go/bin:/usr/local
# note docker on windows requires the bash statement to run this...
bash hugo --gc -b / -d $2 
