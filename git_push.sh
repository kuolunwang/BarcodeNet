#!/bin/bash


if [ ! "$1" ]; then
    echo "commit detail please"
    return
fi
echo "commit: $1"

COMMIT=$1
BRANCH=master

if [ ! -z "$2" ]; then
    echo "operator on branch: $2"
    BRANCH=$2
fi

source git_pull.sh $BRANCH
PULLSTAT=$?
if [ "$PULLSTAT" -gt 0 ] ; then
   echo "There is conflict. Aborting"
   cd ~/BarcodeNet/
   return
fi
echo "-------------------------pull success----------------------------------"

# push pyrobot
echo "-----------------------------------------------------------------------"
echo "-------------------------push carafe----------------------------------"
echo "-----------------------------------------------------------------------"
cd ~/BarcodeNet/model/carafe
git add -A
git commit -m "$1 on core"
git push

# push main
echo "-----------------------------------------------------------------------"
echo "-------------------------push BarcodeNet----------------------"
echo "-----------------------------------------------------------------------"
cd ~/BarcodeNet/
git add -A
git commit -m "$1"
git push 