#!/bin/bash
# arguments:
# 1: git repo
# 2: website destination directory
# 3: website source directory

echo About to generate website
echo ...cloning $1 into $3
rm -r -f $3
git clone $1 $3
cd $3
echo ...cloned, moved to directory: $PWD
echo ...running hugo into $2

export PATH=$PATH:/usr/local/go/bin:/usr/local
# note use absolute path as docker on windows has issues
/usr/local/hugo --gc -b / -d $2
echo ...created
