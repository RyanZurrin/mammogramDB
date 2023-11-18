# Connecting to Rootless Docker within the Chimera13 Compute Node.
Out of the two A:100 DGX compute nodes—Chimera12 and Chimera13—accessible to users of the UMB Gibbs HPC cluster, Chimera13 has rootless Docker set up; this means 
you must follow a set of steps *every time* you desire to run docker on Chimera13.

## Set Home and Runtime Directory Environment Variables
Sign in to your Chimera account and ensure both that you have logged into the Chimera13 node and that you have activated an appropriate conda environment. 
Instructions for accessing Chimera as well as activating an environment can be found [here](https://github.com/mpsych/omama/tree/main/tutorials/chimera%20access).

From within Chimera13, first set your Home environment variable and then set your Runtime Directory environment variable:
```
$ export HOME=/home2/<your.username>
$ export XDG_RUNTIME_DIR=$HOME/.docker/xrd
```

## Ensure Empty Directory for the Directory Environment Variables
You must ensure an empty directory every time you reconnect to rootless docker, therefore simply run the following two commands after setting your Runtime Directory environment variable:
```
$ rootlesskit rm -rf $XDG_RUNTIME_DIR
$ mkdir -p $XDG_RUNTIME_DIR
```

## Set Docker Socket Variable and Run Rootless Docker Install Script
Now set the docker socket variable followed by running the rootless docker install script (we route the output of this script to the background so that you remain in control of the screen session):
```
$ export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
$ dockerd-rootless.sh &>log.out &
```

You may view the output of this script to ensure no errors are outputed, specifically you want to note that docker daemon is listening to a websocket listener; this implies it is connected:
```
tail log.out
```

## Test Docker Connection
Now you are ready to test the docker connection. A quick way to do this is to run the following and note that no error output follows:
```
docker ps
```

You are now ready to run the DeepSight classifier!

## Terminating Running Processes
Though the docker background process should terminate automatically upon your exiting of the compute node, you can manually look for, and terminate, jobs with the following:
```
jobs -l
kill <process id>
```



