#!/bin/bash

source ~/.bash_profile

USERNAME='p.bendiksen001' 
DOCKER='docker'
group=999

id -g $USERNAME
groupadd -g 999 $USERNAME

if id -nG "$USERNAME" | grep -qw "$DOCKER"; then
    echo $USERNAME belongs to $DOCKER
    groups
else
    echo $USERNAME does not belong to $GROUP
fi