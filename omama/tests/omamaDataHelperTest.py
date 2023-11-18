import os
from omama.helpers.data_helper import DataHelper


def test_get2D_cancer():
    length = 20
    img1 = DataHelper.get2D(cancer=True, N=length, config_num=1)
    img2 = DataHelper.get2D(cancer=True, N=length, config_num=1)
    assert len(img1) == length
    assert len(img2) == length
    for i in range(length):
        assert img1[i].label == img2[i].label
        assert img1[i].filePath == img2[i].filePath
        assert img1[i].pixels.all() == img2[i].pixels.all()


def test_get3D_cancer():
    length = 4
    img1 = DataHelper.get3D(cancer=True, N=length, config_num=1)
    img2 = DataHelper.get3D(cancer=True, N=length, config_num=1)
    assert len(img1) == length
    assert len(img2) == length
    for i in range(length):
        assert img1[i].label == img2[i].label
        assert img1[i].filePath == img2[i].filePath
        assert img1[i].pixels.all() == img2[i].pixels.all()


def test_get2D_cancer_random():
    length = 20
    img1 = DataHelper.get2D(cancer=True, N=length, randomize=True, config_num=1)
    img2 = DataHelper.get2D(cancer=True, N=length, randomize=True, config_num=1)
    assert len(img1) == length
    assert len(img2) == length
    for i in range(length):
        assert img1[i].label == img2[i].label
        assert img1[i].filePath != img2[i].filePath


def test_get3D_cancer_random():
    length = 4
    img1 = DataHelper.get3D(cancer=True, N=length, randomize=True, config_num=1)
    img2 = DataHelper.get3D(cancer=True, N=length, randomize=True, config_num=1)
    assert len(img1) == length
    assert len(img2) == length
    for i in range(length):
        assert img1[i].label == img2[i].label
        assert img1[i].filePath != img2[i].filePath


def test_get_method():
    img = DataHelper.get("DXm.2.25.80094907810245643108344883826693135855")
    assert (
        img.filePath == "/raid/data01/deephealth/dh_dcm_ast/"
        "2.25.6626856381473600788855435184358023085/"
        "DXm.2.25.80094907810245643108344883826693135855"
    )


def test_store_method():
    img = DataHelper.get("DXm.2.25.80094907810245643108344883826693135855")
    DataHelper.store(img, "test.dcm")
    assert os.path.exists("test.dcm")
    os.remove("test.dcm")
