# <span style="color:white">QUICK REFERENCE SNIPPETS</a> <a name="TOP"></a>  #

# <span style="color:white">QUICK REFERENCE</a>   #

### Full API avalable in the [Wiki Docs](https://github.com/mpsych/omama/wiki/Omama-API-Documentation) ###

### [Install Environment, yml file here](https://github.com/mpsych/omama/tree/main/tutorials/chimera_access_conda_env) ###
``` python
conda env create -f TF25dcm.yml
```
### Import the Omama package
``` python
import omama O
```
### Get 2D and 3D images ##
by default non-cancer and  N = 1
``` python
# gives 1 nocancer image
img2d = O.DataHelper.get2D()
img3d = O.DataHelper.get3D()
```
To get 10 cancer images
``` python
# gives 10 cancer images
img2d = O.DataHelper.get2D(cancer=True, N=10)
img3d = O.DataHelper.get3D(cancer=True, N=10)
```
To randomize the order of the images:
``` python
# gives 10 non-cancer images in random order
img2d = O.DataHelper.get2D(N=10, randomize=True)
img3d = O.DataHelper.get3D(N=10, randomize=True)
```
### Access image attributes ###
get the image attributes of the first image, works the same for 2D and 3D
``` python
img2d[0].filepath
img2d[0].label
img2d[0].imagelaterality
img2d[0].shape
img2d[0].pixels
```


## Sample uses of get_study ##
#### we can use the StudyInstanceUID to return the data from a study ####
``` python
study = data.get_study(study_instance_uid='2.25.1038...')
```
#### we can use the study_key as well which uses the the global study index location as the key ####
``` python
study = data.get_study(study_key=186945)
```
## How to access the attributes of the get_study ##
### To return the directory path of the study ###
``` python
study.directory
```
> '/r.../d.../dh.../dh_d.../2.25.10387...'
### To return the study instance UID ###
``` python 
study.study_uid
```
> '2.25.10387...'
### To return the study key ###
``` python
study.study_key
```
> 186945
### To return the images in the study ###
``` python
study.images
```
> ['DXm.2.25.7105860005892544414048419613702270984',
'DXm.2.25.34393321181422732362708862077044574162',
'DXm.2.25.199811381614510125916922909435181045283',
'BT.2.25.287417451553593072281164165868824259319',
'BT.2.25.312351113451520848293621641340827348226',
'BT.2.25.25967906531308168040671271408135527531']
### To return additional information about the study (if verbose = True) ###
``` python
study.info
```
> {'total': 6, '3D count': 3, '2D count': 3}
### To return all the information at once ###
``` python
study
```
> will print all the information as stated all at once

## Sample uses of get_images ##
#### We can use the image_id to look up the dicom from the global image list ####
``` python
img = data.get_image(image_id = 44488, dicom_header=True) 
```
#### We can also use the dicom file name like this to get access to the dicom data ####
``` python
img = data.get_image(dicom_name = 'DXm.2.25.1673...')
```
#### Likewise, the path will also work to access the dicom data ####
``` python
img = data.get_image(path='/home/user/data/2.2.25.444.../DXm.2.25.1673...')
```
#### We can specify images within a study by using the get_study method inside the get_image,allowing us to use dot notation to access the images attribute, and then using bracket notation we can then access each image index within the study. ####
``` python 
img = data.get_image(dicom_name=data.get_study('2.25.2653...').images[2]) 
```
## How to access the attributes of the get_image ##
### To return the path to the dicom image ###
``` python
img.filePath
```
 > '/r.../d.../dh.../dh_dcm_ast/2.25.2653.../DXm.2.25.3212...'

### To return the dicom label information ###
``` python
img.label
``` 
> 'NonCancer'

### To return the image laterality ###
``` python
img.imageLaterality
```
> 'L'

### To return the shape of the image ###
``` python
 img.shape 
```
> (3062, 2394)

### To return the pixel array of the image (if pixels=True) ###
``` python
img.pixels
```

>array([[   0,    0,    0, ...,    0,    0,    0],<br/>
        &emsp;&emsp;&emsp;[   0,    0,    0, ...,    0,    0,    0],<br/>
        &emsp;&emsp;&emsp;[   0,    0,    0, ...,    0,    0,    0],<br/>
        &emsp;&emsp;&emsp;...,<br/>
        &emsp;&emsp;&emsp;[2702, 2731, 2687, ..., 1450, 1446, 1455],<br/>
        &emsp;&emsp;&emsp;[2739, 2754, 2748, ..., 1613, 1609, 1620],<br/>
        &emsp;&emsp;&emsp;[2815, 2813, 2825, ..., 1782, 1784, 1790]], dtype=uint16))

### To return the full dicom header information (if dicom_header=True) ###
``` python 
img.metadata
```
> Dataset.file_meta -------------------------------<br/>
                   (0002, 0000) File Meta Information Group Length  UL: 192<br/>
                   (0002, 0001) File Meta Information Version       OB: b'\x00\x01'<br/>
                   (0002, 0002) Media Storage SOP Class UID         UI: Digital Mammography X-Ray Image Storage - For Presentation<br/>
                   (0002, 0003) Media Storage SOP Instance UID      UI: 2.25.303711300619532053936661050692972282983<br/>
                   (0002, 0010) Transfer Syntax UID                 UI: JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])<br/>
                   (0002, 0012) Implementation Class UID            UI: 1.2.276.0.7230010.3.0.3.6.4<br/>
                   (0002, 0013) Implementation Version Name         SH: 'OFFIS_DCMTK_364'<br/>
                   ------------------------------------------------- <br/>
                   ... # cut short becuase dicom headers are very large<br/>


## sample use of next_image to create a generator ## 
#### This is a  generator that will save all the 2d cancer labeled images into a list ####
``` python
img = []
generator = data.next_image(_2d = True, label = Label.CANCER)
for i in range(data.total_2d_cancer):
    img.append(next(generator))
```
#### This is a generator that will save all the 3d non-cancer labeled images into a list ####
``` python
img = []
generator = data.next_image(_3d = True, label = Label.NONCANCER)
for i in range(data.total_3d_cancer):
    img.append(next(generator))
````
#### This is a generator that will save 100 the 2d preindex cancer images into a list ####
``` python
img = []
generator = data.next_image(_2d = True, label = Label.PREINDEX)
for i in range(100):
    img.append(next(generator))
```
## How to access the attributes of each image that was generated ##
### To return the path to the dicom image ###
``` python
img[0].filePath
```
> '/r.../d.../dh.../dh_dcm_ast/2.25.2653.../DXm.2.25.3212...'
### To return the dicom label information ###
``` python
img[0].label
``` 
> 'NonCancer'
### To return the image shape ###
``` python
img[0].shape
```
> (3062, 2394)
### To return the image pixel array ###
``` python
img[0].pixels
```
> array([[   0,    0,    0, ...,    0,    0,    0],<br/>
        &emsp;&emsp;&emsp;[   0,    0,    0, ...,    0,    0,    0],<br/>
        &emsp;&emsp;&emsp;[   0,    0,    0, ...,    0,    0,    0],<br/>
        &emsp;&emsp;&emsp;...,<br/>
        &emsp;&emsp;&emsp;[2702, 2731, 2687, ..., 1450, 1446, 1455],<br/>
        &emsp;&emsp;&emsp;[2739, 2754, 2748, ..., 1613, 1609, 1620],<br/>
        &emsp;&emsp;&emsp;[2815, 2813, 2825, ..., 1782, 1784, 1790]], dtype=uint16)

[Go To TOP](#TOP)
