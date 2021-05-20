#!/bin/bash

POLICY=store #cache, erase
if [ ! -z "$1" ]; then
    POLICY=$1
fi

git config --global credential.helper $POLICY

echo "policy: $(git config --global credential.helper) username & password "