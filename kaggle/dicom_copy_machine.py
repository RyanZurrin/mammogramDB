import pydicom
import matplotlib.pyplot as plt


class DicomCopyMachine:
    """
    This class will take two lists or two dicoms A and B and copy all the header
    information from A and pixel information from B and make a new dicom C
    and save it as new dicom.
    data on each
    dicom intact.
    """

    def __init__(self, dicom_a, dicom_b, save_path=None):
        """
        Parameters
        ----------
        dicom_a : pydicom.dataset.FileDataset
            The dicom data
        dicom_a : pydicom.dataset.FileDataset
            The dicom data
        save_path : str
            The path to save the new dicom to
        """
        self.dicom_A = dicom_a
        self.dicom_B = dicom_b
        self.save_path = save_path
        self.dicom_C = self.copy_dicom()
        print(f"New dicom saved to {self.save_path}")

    def copy_dicom(self):
        """
        Copy the dicom A header info and replace the pixel data and pixel
        transfer syntax with the pixel data and transfer syntax from
        dicom B. This will use proper encapsulation as the pixel data is compressed.

        Jpeg fromat needs to start with 0xFFD8 and end with 0xFFD9
        Jpeg2000 needs to start with 0xFF4F and end with 0xFFD9 so make sure
        this is the case.

        Returns
        -------
        new_dicom : pydicom.dataset.FileDataset
            The new dicom
        """
        new_dicom = self.dicom_A
        new_dicom.PixelData = self.dicom_B.PixelData

        # TODO: crashes kernal, figure out if there is other way to interpret
        #  pixel data properly
        # new_dicom.file_meta.TransferSyntaxUID = \
        #     self.dicom_B.file_meta.TransferSyntaxUID

        # add the window center and width from B to C
        # new_dicom.WindowCenter = self.dicom_B.WindowCenter
        # new_dicom.WindowWidth = self.dicom_B.WindowWidth

        if self.save_path is not None:
            new_dicom.save_as(self.save_path)
        return new_dicom

    def view_dicom(self):
        """
        View the dicom using matplotlib
        """
        plt.imshow(self.dicom_C.pixel_array, cmap=plt.cm.bone)
        plt.show()
