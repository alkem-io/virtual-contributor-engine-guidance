#!/bin/bash
rm -r -f ~/alkemio/website-source 
git clone $1 ~/alkemio/website-source
cd ~/alkemio/website-source
export PATH=$PATH:/usr/local/go/bin:/usr/local
/usr/local/hugo --gc -b / -d $2 
