# <span style="color:white">DATA QUICK REFERENCE</a>   #

### Full API avalable in the [Wiki Docs](https://github.com/mpsych/omama/wiki/Omama-API-Documentation) ###

### [Install Environment, yml file here](https://github.com/mpsych/omama/) ###
``` python
conda env create -f environment.yml
```
### Import the Omama package
``` python
import omama as O
```
### Get 2D and 3D images ###
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
### Get a single image ###
``` python
img = O.DataHelper.get("image_name or path here")
```

### Store an image ###
``` python
O.DataHelper.store(img, "path/to/store/image.dcm")
```

### view an image ###
will display the image. If it is 3D will display the center frame by default.
``` python
img = O.DataHelper.get("image_name or path here")
O.DataHelper.view(img)
```
To view a specific frame:
``` python
O.DataHelper.view(img, frame=22)
```

### view a list of images in a grid ###
You can pass in either 2D or 3D images or a list containing both.
``` python
img2d = O.DataHelper.get2D(N=10, randomize=True)
img3d = O.DataHelper.get3D(N=10, randomize=True)
all_imgs = img2d + img3d
O.DataHelper.view_grid(all_imgs)
```
You can specify the number of columns as well:
``` python
O.DataHelper.view_grid(all_imgs, ncols=3)
```
You can also view a specific frame in the 3D images:
``` python
O.DataHelper.view_grid(all_imgs, frame=22)
```
