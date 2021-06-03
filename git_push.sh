#!/bin/bash

current_path=$(pwd)
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
   cd $current_path/
   return
fi
echo "-------------------------pull success----------------------------------"

# push main
echo "-----------------------------------------------------------------------"
echo "-------------------------push BarcodeNet-------------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/
git add -A
git commit -m "$1"
git push 