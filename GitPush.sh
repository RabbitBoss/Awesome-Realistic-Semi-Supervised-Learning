#!/bin/bash

echo "happy GitAutoPush Starting..."
time=$(date "+%Y-%m-%d %H:%M:%S")
git add .

read -t 30 -p "Please enter a commit comment:" msg

if  [ ! "$msg" ] ;then
    echo "[commit message] "
	git commit -m ""
else
    echo "[commit message] $msg"
	git commit -m "$msg"
fi


git push
echo " GitAutoPush Ending..."
