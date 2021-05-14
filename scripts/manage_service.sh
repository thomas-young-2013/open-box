#!/bin/bash
case $1 in                                        
   start) cd `dirname $0`/..; nohup python litebo/artifact/manage.py runserver 0.0.0.0:11425 &;;
   stop) netstat -nlpt | grep 11425 | awk -F ' ' '{print $7}' | cut -d / -f 1 | xargs kill -9;;
   *) echo "require start|stop" ;;
esac
