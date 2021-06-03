#! /bin/bash

current_path=$(pwd)
BRANCH=package
if [ ! -z "$1" ]; then
    echo "pull branch: $1"
    BRANCH=$1
fi

echo "-----------------------------------------------------------------------"
echo "-------------------------pull BarcodeNet-------------------------------"
echo "-----------------------------------------------------------------------"
git pull

CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in BarcodeNet. Aborting"
   return 1
fi

BRANCH=master

echo "-----------------------------------------------------------------------"
echo "-------------------------pull carafe-----------------------------------"
echo "-----------------------------------------------------------------------"
cd $current_path/barcode_detection/src/model/carafe
git checkout $BRANCH
git pull

CONFLICTS=$(git ls-files -u | wc -l)
if [ "$CONFLICTS" -gt 0 ] ; then
   echo "There is conflict in carafe. Aborting"
   return 1
fi

cd $current_path
return 0