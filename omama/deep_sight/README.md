# <span style="color:white">DEEPSIGHT QUICK REFERENCE</a>   #

### Full API avalable in the [Wiki Docs](https://github.com/mpsych/omama/wiki/Omama-API-Documentation) ###

## Import the Omama package
``` python
import omama as O
```
## Using DeepSight API to run the classifier ##
Running from passing list of images from the DataHelper class:
``` python
# get the images
img2d = O.DataHelper.get2D(cancer=True, N=10)
img3d = O.DataHelper.get3D(cancer=True, N=10)
imgs = img2d + img3d
predictions = O.DeepSight.run(imgs)
```

Running from passing a single image path:
``` python
prediction = O.DeepSight.run("path/to/image.dcm")
```

Running from passing a list of image paths:
``` python
list_of_images = ["path/to/image1.dcm", "path/to/image2.dcm", "path/to/image3.dcm"]
predictions = O.DeepSight.run(list_of_images)
```

Running from passing in a generator from the Data class:
``` python
# get the images
omama_loader = O.OmamaLoader()
data = O.Data(omama_loader)
generator = data.next_image(_2d=True, _3d=True, cancer=True, randomize=True)
predictions = O.DeepSight.run(generator)
```

Running from passing in the Data class object:
``` python
omama_loader = O.OmamaLoader()
data = O.Data(omama_loader)
predictions = O.DeepSight.run(data)
```

Running from passing in a caselist file path:
``` python
predictions = O.DeepSight.run("path/to/caselist.txt")
```

### Sample predictions ###
>{<br/>
'2.25.65612311228836980623594538541927661345': {'coords': None,<br/>
  'score': -1,<br/>
  'errors': ['FAC-170: modified view',<br/>
   'HOL-10: image type',<br/>
   'HOL-20: estimated radiographic magnification factor',<br/>
   'HOL-30: photometric interpretation',<br/>
   'HOL-50: bits stored',<br/>
   'HOL-60: high bit',<br/>
   'HOL-110: grid',<br/>
   'HOL-140: rows',<br/>
   'HOL-150: columns',<br/>
   'HOL-180: pixel intensity relationship',<br/>
   'HOL-190: pixel intensity relationship sign',<br/>
   'HOL-200: presentation lut shape',<br/>
   'HOL-170: window width']},<br/>
 '2.25.25676143079212609526398396217373041691': {'coords': [367.0,<br/>
   1331.0,<br/>
   680.0,<br/>
   1674.0],<br/>
  'score': 0.8718476295471191,<br/>
  'errors': None},<br/>
 '2.25.271328977313727068746568537830916288883': {'coords': [1138.0,<br/>
   1614.0,<br/>
   1873.0,<br/>
   2010.0],<br/>
  'score': 0.8751251101493835,<br/>
  'errors': None},<br/>
 '2.25.135165624907069169803736429415703757935': {'coords': [1858.0,<br/>
   1820.0,<br/>
   1912.0,<br/>
   1943.0],<br/>
  'score': 0.4502192735671997,<br/>
  'errors': None}<br/>
 }<br/>

## Using DeepSight API to get logs ##
Return a dictionary of all the logs
``` python
logs = O.DeepSight.get_logs()
```

Return a dictionary of logs filtered by username
``` python
logs = O.DeepSight.get_logs(username="username")
```

Return a dictionary of logs filtered by date
``` python
logs = O.DeepSight.get_logs(date="date")
```

Return a dictionary of logs filtered by username and date
``` python
logs = O.DeepSight.get_logs(username="username", date="date")
```
### Sample logs ###
>{<br/>
'log_0': {'user_name': 'p.bendiksen001',<br/>
  'date_time': '2022-05-26 14:26:48.319509',<br/>
  'total_time': '84.40804934501648',<br/>
  'current_path': '/tmp/test/deepsightOut/6f8972f83a064460bae80bf3d8b8a68e/'},<br/>
 'log_1': {'user_name': 'ryan.zurrin001',<br/>
  'date_time': '2022-05-31 08:35:03.790876',<br/>
  'total_time': '82.5755684375763',<br/>
  'current_path': '/tmp/test/deepsightOut/00a7c6b709f34915b3e4ef7fa8b6f768/'},<br/>
 'log_2': {'user_name': 'ryan.zurrin001',<br/>
  'date_time': '2022-05-31 11:09:20.662209',<br/>
  'total_time': '83.05247735977173',<br/>
  'current_path': '/tmp/test/deepsightOut/9b6fe0aa8d324dafa9d9881acf9e8d84/'},<br/>
 'log_3': {'user_name': 'p.bendiksen001', <br/>
  'date_time': '2022-05-26 14:30:35.204737', <br/>
  'total_time': '84.69259929656982', <br/>
  'current_path': '/tmp/test/deepsightOut/5c4c37e56f6f458fb5da0df5a7a435ed/'} <br/>
 }<br/>

## Using DeepSight API to get predictions ##
Return a dictionary of all the predictions from a file
``` python
predictions = O.DeepSight.get_predictions("5c4c37e56f6f458fb5da0df5a7a435ed")
```