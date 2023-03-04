#!/bin/bash

echo "happy GitAutoPush Starting..."
time=$(date "+%Y-%m-%d %H:%M:%S")
git add .

read -t 30 -p "请输入提交注释:" msg

if  [ ! "$msg" ] ;then
    echo "[commit message] "
	git commit -m ""
else
    echo "[commit message] $msg"
	git commit -m "$msg"
fi


git push
echo " GitAutoPush Ending..."
