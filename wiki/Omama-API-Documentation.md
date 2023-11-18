<a name="TOP"></a>
The **Omama** API defines a set of python classes and modules that are used for accessing mammograms which are stored as Dicom files. These tools have been developed for the specifics of 2D and 3D mammogram modalities. The classes and features of the Omama modules will allow for easy exploration of Dicom data allowing researchers to gain a better understanding of the data they have. This is a very important step in the early stages of data exploration and can help individuals to better decide what data should be used when building machine learning and artificial intelligent breast cancer detection models.
- - - -
## <span style="color:white">Getting Started</a> ##
<a name="DATA_USE"></a>
You will need to make sure you have pydicom installed. If you have followed the conda environment setup instructions in the 
[Tutorials](https://github.com/mpsych/omama/tree/main/tutorials/chimera_access_conda_env) section and used the TF25dcm.yml
file to create a conda environment, you will have pydicom installed already. If you do not have pydicom installed you will
need to install it by running the following command:

``` python
conda install -c conda-forge pydicom=2.0.0
```
or
``` python
pip install pydicom=2.0.0
```
In this example, we will use our OmamaLoader to load the Data Object. The OmamaLoader is a child class of the DataLoader which 
is used to tell the Data class where to find the data files. To follow this example you will need to have a directory named
omama that contains the following files: data.py, omama_loader.py, data_loader.py, and __init__.py. be sure the
__init__.py file is in the root of the omama directory and contains the following:

``` python
from .data import Data
from .data_loader import DataLoader
from .omama_loader import OmamaLoader
```
Creating your first Data Object is simple but takes time to load the label data. Be sure to set the cache to True to save
your data to cache for easy loading on future runs.
You will need to open an interactive Jupyter Notebook that is connected to the omama directory. Inside the notebook
run the following commands to load the Data Object:
``` python
import omama as O
from omama.omama_loader import Label

omama_loader = O.OmamaLoader()
data = O.Data(data_loader=omama_loader, cache=True)
```
Loading data from a previously stored cache is as simple as:
``` python
data = O.Data(data_loader=omama_loader, load_cache=True)
```

# <span style="color:white">API Documentation</a> #





# <span style="color:white">Module data_helper </a> #
## *DataHelper -> class of static methods* ##
With use of the DataHelper class you do not need to create a Data Object to access the data.
Instead, you can simply call the DataHelper class and use its methods to access the data.
## Methods: ##
## get2D -> list<br/> ##
``` python
@staticmethod
def get2D(
     N=1,
     cancer=False,
     randomize=False, 
     timing=False
):
```
### Parameters : ###
***N : int, optional<br/>***
> (default is 1) Number of images to return

***cancer : bool, optional<br/>***
> (default is False) If true will return only cancerous images, else will return
> only non-cancerous images 

***randomize : bool, optional<br/>***
> (default is False) If true will return the images in a random order

***timing : bool, optional<br/>***
> (default is False) If true will time execution of method, else will not

***Return : list<br/>***
>   N 2D images from the dataset

## get3D -> list<br/> ##
``` python
@staticmethod
def get3D(
     N=1,
     cancer=False,
     randomize=False, 
     timing=False
):
```
### Parameters : ###
***N : int, optional<br/>***
> (default is 1) Number of images to return

***cancer : bool, optional<br/>***
> (default is False) If true will return only cancerous images, else will return
> only non-cancerous images 

***randomize : bool, optional<br/>***
> (default is False) If true will return the images in a random order

***timing : bool, optional<br/>***
> (default is False) If true will time execution of method, else will not

***Return : list<br/>***
>   N 3D images from the dataset

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

## view -> None (view method)<br/> ##
``` python
@staticmethod
def view(
    image,
    frame_number = None,
    timing=False
):
```
### Parameters : ###
***image : int, str, img<br/>***
> can pass as argument the returned image from any of the DataHelpers static methods  get2D, get3D, and get, or the returned obj's from the Data classes get_image or next_image, as well you can pass in a Dicom images name as a string, or the full path which is also a string, or if you have the image id which is an int, that will work as well to view the image.

***frame_number : int, optional<br/>***
> (default is None) If you pass in a 3D image, you can pass in a frame number to view that frame.

***timing : bool, optional<br/>***
> (default is False) If true will time execution of method, else will not

### sample use of view() to look at a Dicom ### 
``` python
O.DataHelper.view(dicom_name="DXm.2.25.232...")
```
> ![image of Dicom](https://github.com/mpsych/omama/blob/main/omama/WIKI_IMAGES/img.png)

## view_grid -> None (view method)<br/> ##
``` python
@staticmethod
def plot_image_grid (
    images,
    frame_number = None,
    ncols=None, 
    cmap='gray'
):
```
### Parameters : ###
***images : list<br/>***
> List of images to plot

***frame_number : int, optional<br/>***
> (default is None) If you pass in a 3D image, you can pass in a frame number to view that frame.

***ncols : int, optional<br/>***
> (default is None) Number of columns in the grid

***cmap : str, optional<br/>***
>  (default is 'gray')  Color map to use

***timing : bool, optional<br/>***
> (default is False) If true will time execution of method, else will not

### sample use of plot_image_grid() to look at a grid of multiple Dicoms ### 
``` python
# Get a random selection of 72 Dicoms
img_rand = DataHelper.get2D(N=72, cancer=True, randomize=True)
# Plot the images in a grid
O.DataHelper.plot_image_grid(img_rand, _2d=True)
```
> ![image of Dicom](https://github.com/mpsych/omama/blob/main/omama/WIKI_IMAGES/grid_view.png)

## get -> wrapper around Data.get_image<br/> ##
``` python
@staticmethod
def get(
    image, 
    view=False, 
    timing=False,
):
```
### Parameters : ###
***image : str, int, img***
> can pass as argument the returned image from any of the DataHelpers static methods  
> get2D, get3D, and get, or the returned obj's from the Data classes get_image or next_image, 
> as well you can pass in a Dicom images name as a string, or the full path which 
> is also a string, or if you have the image id which is an int, that will work 
> as well to view the image.

***view : bool, optional<br/>***
> (default is False) If true will view the image

***timing : bool, optional<br/>***
> (default is False) If true will time execution of method, else will not

### sample use of get() to get image instance ###
using dicom name to get image instance
``` python
img = O.DataHelper.get("DXm.2.25.232...")
```

## store -> (saves image to local directory)<br/> ##
``` python
@staticmethod
def store(
    image, 
    filename, 
    timing=False
):
```
### Parameters : ###
***image : Data.get_image***
> can pass as argument the returned image from any of the DataHelpers static methods
> get2D, get3D, and get, or the returned obj's from the Data classes 
> get_image or next_image.

***filename : str***
> name of the file to save the image to

***timing : bool, optional<br/>***
> (default is False) If true will time execution of method, else will not

### sample use of store() to save a Dicom ###
``` python
img = O.DataHelper.get("DXm.2.25.232...")
O.DataHelper.store(img, "image1.dcm")
```

# <span style="color:white">Module data_loader</a> #

## *DataLoader* ##
The DataLoader class is used to hold all the specific path location data that is 
used in the data class. You will need to populate this class or build a subclass 
of the DataLoader superclass. We have built such a class called OmamaLoader which
you can find below as an example of how to create this derived class. Once this is 
done you will use that in the Data class. We also give an example of these steps 
in the Data Module documentation below.
```
class DataLoader(
    data_paths: list,
    csv_paths: list,
    study_folder_paths: list,
    pickle_path: str,    
    dicom_2d_substring: str,
    dicom_3d_substring: str
    dicom_tags: list = None, 
    )
```
### Constructor arguments: ###
***data_paths : list, required <br/>***
> A list of strings that are the paths to the individual studies being explored

***csv_paths : list, required <br/>***
> A list of strings that are the paths to the CSV files that contain labels. (FORMAT CSV AS FOLLOWS:
COl1:"StudyInstanceUID", COL2:"ImageLaterality", COL3:"Label", where Label is one of the following:
IndexCancer, PreIndexCancer, or None. If you have a CSV that contains additional data please modify
CSV files to follow this format to use the Data class with your data if you desire to do so.

***study_folder_names : str, required <br/>***
> A list of strings that are the names of the individual studies being explored

***pickle_path : str, required <br/>***
> A path to the pickle file which is used for saving and loading cache 

***dicom_2d_substring : str, required <br/>***
> A common identifying substring, preferably the starting few characters should all be common for every 2d Dicom, which is used as the identifier for partitioning and separating the 2d data

***dicom_3d_substring : str, required <br/>***
> A common identifying substring, preferably the starting few characters should all be common for every 3d Dicom, which is used as the identifier for partitioning and separating the 3d data

***dicom_tags : list, None <br/>***
> (default is None, full Dicom will be read while getting pixel data) Here we can use specific tags for reading pixel data of a Dicom. This is used to limit the data being read from a Dicom while getting only the pixel data.

### Class Instance Variables: ###
var csv_paths
> Returns the csv paths

var data_paths
> Returns the data paths

var dicom_2d_substring
> Returns the dicom 2d substring

var dicom_3d_substring
> Returns the dicom 3d substring

var dicom_tags
> Returns the dicom tags

var pickle_path
> Returns the pickle path

> var study_folder_names
> Returns the study folder names

- - - - 
# <span style="color:white">Module omama_loader</a> #

## [*OmamaLoader*](#OMAMA_SAMPLE_CODE) ##
The OmamaLoader class is a subclass of the DataLoader superclass and holds 
information that is specific to the Omama dataset stored at our compute cluster
location. We will show an example of how to use the superclass to build a child
class of your own. All you should have to change if you are using the Omama 
dataset is the location of where you are storing your data. If you are using your
mammogram Dicom data you will need to make sure your CSV file is formatted as 
specified in the DataLoader class.
### EXAMPLE CODE FOR USING DATALOADER SUPERCLASS <a id="OMAMA_SAMPLE_CODE"></a> ###
```
from omama.data_loader import DataLoader
from types import SimpleNamespace

label_types = {'CANCER': 'IndexCancer', 'NONCANCER': 'NonCancer', 'PREINDEX': 'PreIndexCancer'}
Label = SimpleNamespace(**label_types)


class OmamaLoader(DataLoader):
    def __init__(self):
        data_paths = [r'/raid/data01/deephealth/dh_dh2/',
                      r'/raid/data01/deephealth/dh_dh0new/',
                      r'/raid/data01/deephealth/dh_dcm_ast/']

        csv_paths = [r'/raid/data01/deephealth/labels/dh_dh2_labels.csv',
                     r'/raid/data01/deephealth/labels/dh_dh0new_labels.csv',
                     r'/raid/data01/deephealth/labels/dh_dcm_ast_labels.csv']
                     
        study_folder_names = ['dh_dh2', 'dh_dh0new', 'dh_dcm_ast']

        pickle_path = r"../../../../../shared_home/scratch01/abc/some_folder/pickle/labelData"

        dicom_tags = ["SamplesPerPixel", "PhotometricInterpretation", "PlanarConfiguration", "Rows", "Columns",
                      "PixelAspectRatio", "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
                      "SmallestImagePixelValue", "LargestImagePixelValue", "PixelPaddingRangeLimit",
                      "RedPaletteColorLookupTableDescriptor", "GreenPaletteColorLookupTableDescriptor",
                      "BluePaletteColorLookupTableDescriptor", "RedPaletteColorLookupTableData",
                      "GreenPaletteColorLookupTableData", "BluePaletteColorLookupTableData", "ICCProfile",
                      "ColorSpace", "PixelDataProviderURL", "ExtendedOffsetTable", "NumberOfFrames",
                      "ExtendedOffsetTableLengths", "PixelData"]

        dicom_2d_substring = "DXm"

        dicom_3d_substring = "BT"
        super().__init__(
              data_paths,
              csv_paths,
              pickle_path, 
              study_folder_names, 
              dicom_tags, 
              dicom_2d_substring, 
              dicom_3d_substring
        )

```

- - - - 
# <span style="color:white">Module data</a> #

## [*Data*](#DATA_USE) ##
The Data class is used to explore Dicom data and the related label data. It then generates basic statistical counts of things such as how many 2d and 3d images there are, and within each of those groups, additional counts like cancer, noncancer, and pre-index cancer counts are generated. This type of information is helpful when trying to understand the data better. The Data class is composed of three main public methods: *get_study*, which allows one to look up studies easily; *get_image*, which allows one to look up image data and *next_image*, which returns a python generator of images which fit the filtered flags specified. These methods make it easy to access the Dicom files which can sometimes be difficult to explore and manage. The Omama modules will help anyone who plans to use this dataset giving them fast and easy access to the data they plan to use.
```
class Data(
    data_loader: data_loader.DataLoader, 
    cache: bool = True, 
    load_cache: bool = False, 
    print_data: bool = False, 
    timing: bool = False
    )
```
### Constructor arguments: ###
***data_loader : DataLoader, required <br/>***
> This is an instance of the DataLoader class or you can use inheritance to build a child class of the DataLoader and populate that with the specific information on where to find data, CSV files as well as how to determine other specifics to the dataset one is working with.

***cache : bool <br/>***
> (default True) Whether to cache the data. If this is true, it will generate fresh label information which is stored in a map and then saved into a pickle file. The pickle file path needs to be set and configured in the DataLoader class before specifying this as true or else it will just cache the file in the working directory with a default name of cache data. If it is false and you are not loading from a cache file then the data will not be saved that is generated. The generation process can take up to an hour or more depending on the data you are generating labels from.

***load_cache : bool <br/>***
> (default False) Whether to load the data from the cache. If this is true it will load the label information into the Data class instance and use that instead of generating new label information. The pickle path needs to be configured in the DataLoader before this will load properly.

***print_data : bool, optional <br/>***
> (default is False) Prints out a detailed dictionary of information about all the studies and the combined data upon completing its initialization process

***timing : float, optional <br/>***
> (default is False) Sets all the timers in the initializing methods to true, and prints out how long each process took. This is good for debugging and finding bottle-necks in the processes. 


### Attributes: ###
***total_2d_noncancer : int <br/>***
> total number of 2D dicoms with a label of NonCancer

***total_2d_cancer : int <br/>***
> total number of 2D dicoms with a label of IndexCancer

***total_2d_preindex : int <br/>***
> total number of 3D dicoms with a label of PreIndexCancer

***total_3d_noncancer : int <br/>***
> total number of 3D dicoms with a label of NonCancer

***total_3d_cancer : int <br/>***
> total number of 3D dicoms with a label of IndexCancer

***total_3d_preindex : int <br/>***
> total number of 3D dicoms with a label of PreIndexCancer

## Methods: ##

## [get_study -> dictionary](#GET_STUDY_SAMPLE_CODE)<br/> ##
Get a study from either the StudyInstanceUID or study key. study key is derived from a sorted list of studies and is the index of where the particular study is saved in the list
```
@classmethod
def get_study(
    cls, 
    study_instance_uid: str = None,
    study_key: int = None,
    verbose=False,
    timing=False
    )
```
### Parameters : ###
***study_instance_uid : str, optional<br/>***
> (default is None, so either this or the ID needs to be set)
Uses the StudyInstanceUID to locate the study information and returns data as dictionary 

***study_key : int, optional<br/>***
> (default is None, so either this or the study_instance_uid needs to be set) Uses the location of a study which is based on its index in the sorted list of all studies

***verbose : bool, optional <br/>***
> (default is False) If this is set to True it will also include some additional information about the study such as the number of 3D and 2D Dicoms in the study

***timing : bool, optional <br/>***
> (default is False) If true will time execution of method, else will not

***Return : dictionary<br/>***
>  a dictionary of data about the specified study

<a name="GET_STUDY_SAMPLE_CODE"></a>
### Sample uses of get_study ###
#### we can use the StudyInstanceUID to return the data from a study ####
``` python
study = data.get_study(study_instance_uid='2.25.1038...')
```
#### we can use the study_key as well which uses the the global study index location as the key ####
``` python
study = data.get_study(study_key=186945)
```
### How to access the attributes of the get_study ###
#### To return the directory path of the study ####
``` python
study.directory
```
> '/r.../d.../dh.../dh_d.../2.25.10387...'
#### To return the study instance UID ####
``` python 
study.study_uid
```
> '2.25.10387...'
#### To return the study key ####
``` python
study.study_key
```
> 186945
#### To return the images in the study ####
``` python
study.images
```
> ['DXm.2.25.7105860005892544414048419613702270984',
'DXm.2.25.34393321181422732362708862077044574162',
'DXm.2.25.199811381614510125916922909435181045283',
'BT.2.25.287417451553593072281164165868824259319',
'BT.2.25.312351113451520848293621641340827348226',
'BT.2.25.25967906531308168040671271408135527531']
#### To return additional information about the study (if verbose = True) ####
``` python
study.info
```
> {'total': 6, '3D count': 3, '2D count': 3}
#### To return all the information at once ####
``` python
study
```
> will print all the information as stated all at once


## [get_image -> np.array, dictionary](#GET_IMAGES_SAMPLE_CODE) <br/>  ##
Gets the pixels of an image as well as some additional data as specified
```
@classmethod
def get_image(
    cls,
    image_id: int = None,
    dicom_name: str = None,
    path: str = None, 
    pixels: bool = True,
    dicom_header: bool = False,
    timing=False
    )
```
### Parameters : ###
***image_id : int, optional<br/>*** 
> (default is None, either this, dicom_id or path needs to be set) Uses the image_id to find the Dicom information and returns the specified data from the image

***dicom_name : str, optional<br/>*** 
> (default is None, either this, dicom_id, or path needs to be set) Uses the dicom_name which is the file name of the Dicom, to locate the Dicom information and returns the specified data from the image

***path : str, optional<br/>*** 
> (default is None, so either this, image_id, or dicom_name needs to be set) Uses the path of a Dicom image to process it and returns the specified data from the image 

***pixels : bool, optional<br/>***
> (default is True) If True will return a NumPy array containing all the pixel data of the Dicom, else will return None 

***dicom_header : bool, optional<br/>***
> (default is False) If True will return the dicom header as a dictionary, else will return None 

***timing : bool, optional<br/>***
> (default is False) If true will time execution of method, else will not  

***Return :<br/>***
> NumPy Array, dictionary \[np.array just the pixels or None if pixels=False, info] info is dictionary: {'label': LABELS.Cancer, 'filepath':… 'shape'…}

<a name="GET_IMAGES_SAMPLE_CODE"></a>
### Sample uses of get_images ###
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
### How to access the attributes of the get_image ###
#### To return the path to the dicom image ####
``` python
img.filePath
```
 > '/r.../d.../dh.../dh_dcm_ast/2.25.2653.../DXm.2.25.3212...'

#### To return the dicom label information ####
``` python
img.label
``` 
> 'NonCancer'

#### To return the image laterality ####
``` python
img.imageLaterality
```
> 'L'

#### To return the shape of the image ####
``` python
 img.shape 
```
> (3062, 2394)

#### To return the pixel array of the image (if pixels=True) ####
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

#### To return the full dicom header information (if dicom_header=True) ####
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



## [next_image -> np.array, dictionary](#NEXT_IMAGE_SAMPLE_CODE)<br/> ##
Generator to filter and return iteratively all the filtered images in the dataset.
```
@classmethod
def next_image(
    cls,
    _2d: bool = False,
    _3d: bool = False,
    label: Label = None,
    randomize: bool = False,
    timing: bool = False
    )
```
### Parameters : ###
***_2D : int, optional<br/>*** 
> (default is None, so either this or the _3D or both needs to be set) Filter flag used to add 2D images to returned set of Dicoms 

***_3D : str, optional<br/>*** 
> (default is None, so either this or the _2D or both needs to be set) Filter flag used to add 3D images to returned set of Dicoms 

***label : str, optional<br/>*** 
> (default is None which is all images, additional options are IndexCancer, NonCancer, PreIndexCancer) Uses the path of a Dicom image to process it and return the pixel data as NumPy array if pixels is True

***randomize : bool, optional<be/>***
> (default is False) If True will randomize the order of the returned images

***timing : bool, optional<br/>*** 
> (default is False) If true will time execution of method, else will not

***Return :<br/>***
> NumPy Array, dictionary \[np.array just the pixels, info] info is dictionary: {'label': LABELS.Cancer, 'filepath':… 'shape'…}

<a name="NEXT_IMAGE_SAMPLE_CODE"></a>
### sample use of next_image to create a generator ### 
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
for i in range(data.total_3d_noncancer):
    img.append(next(generator))
````
#### This is a generator that will save 100 the 2d preindex cancer images into a list ####
``` python
img = []
generator = data.next_image(_2d = True, label = Label.PREINDEX)
for i in range(100):
    img.append(next(generator))
```
### How to access the attributes of each image that was generated ###
#### To return the path to the dicom image ####
``` python
img[0].filePath
```
> '/r.../d.../dh.../dh_dcm_ast/2.25.2653.../DXm.2.25.3212...'
#### To return the dicom label information ####
``` python
img[0].label
``` 
> 'NonCancer'
#### To return the image shape ####
``` python
img[0].shape
```
> (3062, 2394)
#### To return the image pixel array ####
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


## [filter_data -> Internal modifier](#FILTER_DATA_SAMPLE_CODE)<br/> ##
Filters the dataset to include only Dicoms that match the specified
filter flags. Configures the internal class data in a way that all the Data
methods work the same, but only will be on the filtered data. To reset the 
filtered data to the original dataset, use the reset_data method.
```
@classmethod
def filter_data(
    cls,
    _2d=True,
    _3d=True,
    shapes: list = None,
    row_max: int = None,
    row_min: int = None,
    col_max: int = None,
    col_min: int = None,
    frames_max: int = None,
    frames_min: int = None,
    age_range: list = None,
    labels: list = None,
    timing=False
):
```
### Parameters : ###
***_2D : int, optional<br/>*** 
> (default is True, includes 2D images) Filter flag used to remove 2D images

***_3D : str, optional<br/>*** 
> (default is True, includes 3D images) Filter flag used to remove 3D images

***shapes: list['(X1,Y1)', '(X2,Y2)', ..., (Xn,Yn,Zn)'], optional<br/>*** 
> (default is None, includes all shapes) specify the shapes of the images to include

***row_max: int, optional<be/>***
> (default is None, includes all rows) 
> Filter flag used to remove images with rows greater than the specified value

***row_min: int, optional<be/>***
> (default is None, includes all rows) 
> F Filter flag used to remove images with rows less than the specified value

***col_max: int, optional<be/>***
> (default is None, includes all columns)
> Filter flag used to remove images with columns greater than the specified value
 
***col_min: int, optional<be/>***
> (default is None, includes all columns)
> Filter flag used to remove images with columns less than the specified value

***frames_max: int, optional<be/>***
> (default is None, includes all frames)
> Filter flag used to remove images with frames greater than the specified value

***frames_min: int, optional<be/>***
> (default is None, includes all frames)
> Filter flag used to remove images with frames less than the specified value

***age_range: list[min, max]<be/>***
> (Default is None, include all ages) specify the age range in months to filter images by age

***labels: list<be/>***
> (default is None, includes all labels) specify the labels of the images to include

***timing : bool, optional<br/>*** 
> (default is False) If true will time execution of method, else will not

<a name="FILTER_DATA_SAMPLE_CODE"></a>
### Sample use of filter_data to limit data available in the Data instance ### 
``` python
import omama as O

omama_loader = O.OmamaLoader()
data = O.Data(data_loader=omama_loader)
# Filtering for 2D cancer images with a max row and col size 
data.filter_data(_3d=False, row_max=794, col_max=990, labels=['IndexCancer'])
``` 
sample stats before filtering
>namespace(dh_dh2=namespace(total=22144, 3D=9363, 2D=12781),<br/>
          dh_dh0new=namespace(total=126460, 3D=62736, 2D=63724),<br/>
          dh_dcm_ast=namespace(total=819387, 3D=58, 2D=819329),<br/>
          total_all_dicoms=967991,<br/>
          total_2d_all=895834,<br/>
          total_3d_all=72157,<br/>
          total_2d_cancer=14965,<br/>
          total_2d_preindex=1915,<br/>
          total_2d_noncancer=855763,<br/>
          total_3d_cancer=376,<br/>
          total_3d_preindex=0,<br/>
          total_3d_noncancer=67162,<br/>
          total_cancer=15341,<br/>
          total_preindex=1915,<br/>
          total_noncancer=922925,<br/>
          total_no_label=27810)<br/>

sample stats after filtering
> namespace(dh_dh2=namespace(total=0, 3D=0, 2D=0),<br/>
          dh_dh0new=namespace(total=0, 3D=0, 2D=0),<br/>
          dh_dcm_ast=namespace(total=4261, 3D=0, 2D=4261),<br/>
          total_all_dicoms=3403,<br/>
          total_2d_all=3403,<br/>
          total_3d_all=0,<br/>
          total_2d_cancer=4261,<br/>
          total_2d_preindex=0,<br/>
          total_2d_noncancer=0,<br/>
          total_3d_cancer=0,<br/>
          total_3d_preindex=0,<br/>
          total_3d_noncancer=0,<br/>
          total_cancer=3403,<br/>
          total_preindex=0,<br/>
          total_noncancer=0,<br/>
          total_no_label=0)<br/>

## reset_data -> Internal modifier##
Resets the data to the original data
``` python
def reset_data(cls)
```

# <span style="color:white">Module deep_sight</a> #
## *DeepSight -> wrapper class for the deepsight classifier* ##

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


# <span style="color:white">Unit Testing</a> #
We use pytest to build unit tests for each of the modules in the omama 
module. These tests are used to verify that all the modules are running 
correctly and that there are no errors. This is a growing list of tests that 
together test the full scope of the Data class with its use of the 
DataLoader as well as the DataHelper class and its static methods. If you 
want to perform Unit tests with your data you can do so using the omamaTest as 
a template to build your own Unit tests. 

## Unit testing the Omama Data API ##
1. You will need to have cloned the omama repository
2. you will need to follow the tutorial and install the conda environment TF25dcm.
3. Install pytest using pip install pytest
4. Be sure your conda env is activated and navigate to the omama/omama/ 
   directory that contains the modules to be tested as well as the pytest 
   omamaDatatTest.py and the omamaDataHelperTest.py files.
5. When in correct directory type:
```
pytest
```

This will run all the tests which will take a bit to run so be patient please.
If you make changes or are refactoring any of the omama modules these tests 
should be run frequently to make sure nothing breaks.
    
[Go To TOP](#TOP)