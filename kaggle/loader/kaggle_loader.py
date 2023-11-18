from omama.loaders.data_loader import DataLoader
from types import SimpleNamespace

__author__ = "Ryan Zurrin"

label_types = {"CANCER": "IndexCancer", "NONCANCER": "NoneCancer"}
Label = SimpleNamespace(**label_types)


class KaggleLoader(DataLoader):
    """ """

    def __init__(
        self,
        data_paths=None,
        csv_paths=None,
        study_folder_names=None,
        dicom_tags=None,
        dicom_2d_substring=None,
        dicom_3d_substring=None,
        cache_paths=None,
        patient_identifier=None,
        cancer_identifier=None,
        caselist_path=None,
        config_num=None,
    ):
        """
        Initialize the OmamaLoader class.

        Parameters
        ----------
        data_paths : list: str
            paths to the data
        csv_paths : list: str
            paths to the csv files
        study_folder_names : list: str
            names of the study folders
        dicom_tags : list: str
            tags that are used to extract the data from the dicom files
        dicom_2d_substring : str
            substring that is used to find the 2d dicom files
        dicom_3d_substring : str
            substring that is used to find the 3d dicom files
        cache_paths : list: str
            paths to the pickle files which are used for caching
        config_num : int
            The option number corresponds to a set of paths to specific data.
            config_num options are:
            1: Full Kaggle dataset

        """
        if data_paths is None:
            data_paths = [r"/raid/mpsych/kaggle_mammograms/train_images/"]
        else:
            data_paths = data_paths

        if csv_paths is None:
            csv_paths = [r"/raid/mpsych/kaggle_mammograms/kaggle_train.csv"]
        else:
            csv_paths = csv_paths

        if study_folder_names is None:
            study_folder_names = ["train_images"]
        else:
            study_folder_names = study_folder_names

        if config_num is None:
            config_num = 1
        else:
            config_num = config_num

        caselist_path = caselist_path

        if dicom_tags is None:
            dicom_tags = [
                "SamplesPerPixel",
                "PhotometricInterpretation",
                "PlanarConfiguration",
                "Rows",
                "Columns",
                "PixelAspectRatio",
                "BitsAllocated",
                "BitsStored",
                "InstanceNumber",
                "PatientID",
                "HighBit",
                "PixelRepresentation",
                "SmallestImagePixelValue",
                "LargestImagePixelValue",
                "PixelPaddingRangeLimit",
                "SOPInstanceUID",
                "RedPaletteColorLookupTableDescriptor",
                "GreenPaletteColorLookupTableDescriptor",
                "BluePaletteColorLookupTableDescriptor",
                "RedPaletteColorLookupTableData",
                "GreenPaletteColorLookupTableData",
                "BluePaletteColorLookupTableData",
                "ICCProfile",
                "ColorSpace",
                "PixelDataProviderURL",
                "ExtendedOffsetTable",
                "NumberOfFrames",
                "ExtendedOffsetTableLengths",
                "PixelData",
            ]
        else:
            dicom_tags = dicom_tags

        if dicom_2d_substring is None:
            dicom_2d_substring = ""
        else:
            dicom_2d_substring = dicom_2d_substring

        if dicom_3d_substring is None:
            dicom_3d_substring = "N/A"
        else:
            dicom_3d_substring = dicom_3d_substring

        if patient_identifier is None:
            patient_identifier = "InstanceNumber"
        else:
            patient_identifier = patient_identifier

        if cancer_identifier is None:
            cancer_identifier = "InstanceNumber"
        else:
            cancer_identifier = cancer_identifier

        # initialize the cache for the complete dataset
        if config_num == 1:
            if cache_paths is None:
                cache_paths = [
                    "/raid/mpsych/cache_files/kaggle_label_cache",
                    "/raid/mpsych/cache_files/full_pat_id_cache",
                ]
            else:
                cache_paths = cache_paths

        elif config_num == 2:
            if cache_paths is None:
                cache_paths = [
                    "/raid/mpsych/cache_files/deepsight_kaggleSM_label_cache",
                    "/raid/mpsych/cache_files/deepsight_KaggleSM_pat_id_cache",
                ]
            else:
                cache_paths = cache_paths
            if caselist_path is None:
                caselist_path = r"/raid/mpsych/whitelists/kaggle_caselist.txt"
            else:
                caselist_path = caselist_path

        super().__init__(
            data_paths,
            csv_paths,
            study_folder_names,
            cache_paths,
            dicom_tags,
            dicom_2d_substring,
            dicom_3d_substring,
            patient_identifier,
            cancer_identifier,
            caselist_path,
            config_num=config_num,
        )
