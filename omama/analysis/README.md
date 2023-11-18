# Dicom Data Subsetting
Dicom database, local storage, and filtering codebase; our agreed-upon subsetting pipeline filters out images across various conditions, ending with a dataset that maximizes the number of cancerous images as well as the number of digital breast tomosynthesis (3D) images.

## Reading Database to and from Disk
Instantiate the DataSelector Class in order to initiate omama database scraping, and call its load() method to load the outputed file.
Note: execution terminates if present working directory contains a pickle file of the same name as the outputed file.
```
$ ds=DataSelector(timeit=True)
$ ds.load()
```

## Data Insights
### Image By Label Counts
<img src="https://github.com/mpsych/omama/blob/main/omama/analysis/images/image-label-counts.png" width="500" height="500" />

## Subsetting Diagram
<img src="https://github.com/mpsych/omama/blob/main/omama/analysis/images/diagram-subset.png" width="400" height="600" />

## Observations

Original Dataset:
  Total cases: 967991
    - 2D: 895834
    - 3D: 72157
  Studies: 204,192
  Patients: 202,753

### Observations:  
  - all 3d Study IDs are found in the set of 2d Study IDs; this also applies to Patient IDs.
  - all 3d Cancer Study IDs are found in the set of 3d nonCancer Study IDs; this also applies to Patient IDs.
  - number of Patients is always smaller than the number of Studies.

### Subsetting Steps:  
  <ins>Original Dataset</ins> -> <ins>Missing Label and 'Both' Filtering</ins> -> <ins>Small Shape Size Filtering</ins> -> <ins>Minority Manufacturing
    Filtering</ins> -> <ins>Unique Studies filtering for 2D NonCancer images</ins>.


### After Manufacturer Filtering Step:  

#### Representation 1:

  images: 916035
  studies: 197,785
  patients: 196,653

   - 2D: 
    images: 848,501  
    studies: 197,785  
     - NonCancer  
       images: 836,313  
       studies: 195,593  
       patients: 194,483  
     - Cancer  
       images: 12,188  
       unique studies: 4,507  
       unique patients: 4,486  
       - PreCancer  
         images: 1,820  
         studies: 777 (655 of these Study IDs found in 2D nonCancer) (18 of these found in 2D IndexCancer)  
         patients: 777 (655 of these Patient IDs found in 2D nonCancer) (18 of these found in 2D IndexCancer)  
       - IndexCancer  
         images: 10,368  
         studies: 3,748 (1,660 of these Study IDs found in 2D nonCancer)  
         patients: 3,727 (1,661 of these Study IDs found in 2D nonCancer)  

   - 3D: 
    images: 67,534  
    studies: 16,268 (* all of these Study IDs found in 2D)  
       - NonCancer  
         images: 67,158  
         studies: 16,234 (* all of these are found in 2D NonCancer)  
         patients: 15,137 (* all of these are found in 2D NonCancer)  
       - IndexCancer  
         images: 376  
         studies: 170 (136 of these Study IDs found in 3D nonCancer)  
         patients: 149 (137 of these Study IDs found in 3D nonCancer)  


#### Representation 2
```
                              Total Images: 916,035
                              Total Studies: 197,785
                              Total Patients: 196,653

                                2D Images (848,501 images). 197,785 Studies. 196,653 Patients
                               /         \
                              /           \
                             /             \
                            /               \
                           /                 \
                          /                   \
                         /                     \
                  2D Cancer (12,188) (0)      2D nonCancer (836,313)
                    |                             |
                    |                             |
                  Unique Studies:                 |
                    4,507                         |
                  Unique Patients:                |
                    4,486                         |
                    |                             |
                    /\                            |
                   /  \                           | 
                  /    \                          |
                 /      \                         |
        Pre(1,820)      Index(10,368)          Studies:         
        Studies: (1)    Studies: (3)           195,593  
        777             3,748                  Patients:      
        Patients: (2)   Patients: (4)          194,483
        777             3,727                  


                                3D Images (67,534 images). 16,268 Studies. 15,149 Patients (5)
                               /         \
                              /           \
                             /             \
                            /               \
                           /                 \
                          /                   \
                         /                     \
                  3D Cancer                  3D nonCancer 
                      |                          |
                      |                          | 
                      |                          |
                      |                          |
                Index(376)                     (67,158)        
                Studies: (6)                   Studies:
                170                            16,234     
                Patients: (7)                  Patients:
                149                            15,137
                
  0) (2,315 of these Study IDs found in 2D nonCancer, 2,316 of these Patient IDs found in 2D nonCancer)
  1) (655 of these Study IDs found in 2D nonCancer) (18 of these found in 2D IndexCancer)
  2) (655 of these Patient IDs found in 2D nonCancer) (18 of these found in 2D IndexCancer)
  3) (1,660 of these Study IDs found in 2D nonCancer)
  4) (1,661 of these Study IDs found in 2D nonCancer)
  5) (all of these Study IDs found in 2D, all of these Patient IDs found in 2D)
  6) (136 of these Study IDs found in 3D nonCancer)
  7) (137 of these Patient IDs found in 3D nonCancer)
```

### After Unique Studies filtering - of 2D NonCancer images - Step:

#### Representation 1:

images: 259,081  
  studies: 197,785  
  patients: 196,653  

   - 2D:  
    images: 191,547  
    studies: 181,687  
    patients: 181,666  
     - NonCancer  
       images: 179,359  
       studies: 179,359  
       patients: 179,359  
     - Cancer  
       images: 12,188  
       unique studies: 4,507 (2,179 of these Study IDs found in 2D nonCancer)  
       unique patients: 4,486 (2,179 of these Study IDs found in 2D nonCancer)  
       - PreCancer  
         images: 1,820  
         studies: 777 (655 of these Study IDs found in 2D nonCancer) (18 of these found in 2D IndexCancer)  
         patients: 777 (655 of these Patient IDs found in 2D nonCancer) (18 of these found in 2D IndexCancer)  
       - IndexCancer  
         images: 10,368  
         studies: 3,748 (1,524 of these Study IDs found in 2D nonCancer)  
         patients: 3,727 (1,524 of these Patient IDs found in 2D nonCancer)  
   - 3D:  
    images: 67,534  
    studies: 16,268 (170 of these Study IDs found in 2D)  
    patients: 15,149 (162 of these Patient IDs found in 2D)  
     - NonCancer  
       images: 67,158  
       studies: 16,234 (0 found in 2D NonCancer)  
       patients: 15,137 (13 found in 2D NonCancer)  
     - IndexCancer  
       images: 376  
       studies: 170 (136 of these Study IDs found in 3D nonCancer)  
       patients: 149 (137 of these Study IDs found in 3D nonCancer)  
       
 #### Representation 2:
 ```
                               Total Images: 259,081
                              Total Studies: 197,785
                              Total Patients: 196,653

                                2D Images (191,547 images). 181,687 Studies. 181,666 Patients
                               /         \
                              /           \
                             /             \
                            /               \
                           /                 \
                          /                   \
                         /                     \
                  2D Cancer (12,188)        2D nonCancer (179,359)
                    |                             |
                    |                             |
                  Studies:                        |
                    4,507  (0)                    |
                  Patients:                       |
                    4,486  (0)                    |
                    |                             |
                    /\                            |
                   /  \                           | 
                  /    \                          |
                 /      \                         |
        Pre(1,820)      Index(10,368)          Studies:         
        Studies:        Studies:               179,359  
        777  (1)        3,748  (3)             Patients:      
        Patients:       Patients:              179,359
        777  (2)        3,727  (4)                


                                3D Images (67,534 images). 16,268 Studies. 15,149 Patients (5)
                               /         \
                              /           \
                             /             \
                            /               \
                           /                 \
                          /                   \
                         /                     \
                  3D Cancer                  3D nonCancer 
                      |                          |
                      |                          | 
                      |                          |
                      |                          |
                Index(376)                     (67,158)        
                Studies:                       Studies:
                170  (6)                       16,234  (8)   
                Patients:                      Patients:
                149  (7)                       15,137  (9)
                
  0) (2,179 of these Study IDs found in 2D nonCancer, 2,179 of these Patient IDs found in 2D nonCancer)
  1) (655 of these Study IDs found in 2D nonCancer) (18 of these found in 2D IndexCancer)
  2) (655 of these Patient IDs found in 2D nonCancer) (18 of these found in 2D IndexCancer)
  3) (1,524 of these Study IDs found in 2D nonCancer)
  4) (1,524 of these Patient IDs found in 2D nonCancer)
  5) (170 of these Study IDs found in 2D, 162 of these Patient IDs found in 2D)
  6) (136 of these Study IDs found in 3D nonCancer)
  7) (137 of these Patient IDs found in 3D nonCancer)
  8) (0 of these Study IDs found in 2D NonCancer)
  9) (13 of these Patient IDs found in 2D NonCancer)
```
