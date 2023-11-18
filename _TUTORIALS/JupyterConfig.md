# Jupyter Notebook Use with GPU Accessible Chimera Node.
Here we present steps for a two-hop tunneling from your local machine to the Chimera12 DGXA100 node, in the context of Jupyter Notebook remote access.

## Configuration
Sign in to your Chimera account and ensure activation of an appropriate conda environment. Instructions for the aforementioned can be found [here](https://github.com/mpsych/omama/tree/main/tutorials/chimera%20access).


Ensure Jupyter Notebook is installed in your environment:
```
$ conda install notebook
```
and subsequently generate/access a Jupyter Notebook config file:
``` 
$ jupyter-notebook --generate-config
```
```
$ vi /home/UMBusername001/.jupyter/jupyter_notebook_config.py
```
Finally, uncomment and populate the following fields within the file using your IP address and an arbitrary port number (vi search in command mode with /, press Return to go to found string, press n to go to next ocurrence)):
```
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = YOUR_PORT_NUMBER

```

## Use
From within your conda environment activate Jupyter notebook:
```
$ jupyter notebook
```
 Take note that the resulting printout will provide you a URL, with included token value, to paste and use within a browser instance.
 
From a separate terminal instance of your local machine, run:
 ```
 ssh -tt UMBusername001@chimera.umb.edu -L <YOUR_PORT_NUMBER>:chimera12:<YOUR_PORT_NUMBER> -N
 ```
You may change the 12 to 13 to connect to the other GPU compute node. 

Now you may simply paste and use the aforementioned URL to quickly access the Jupyter Notebook IDE or, alternatively, you can type the following into your browser:
```
http://localhost:<YOUR_PORT_NUMBER>
```
wherein you will be prompted to enter the previously mentioned token value.
