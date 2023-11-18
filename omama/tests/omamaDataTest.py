import omama.data as O
import omama.loaders.omama_loader as OL
from omama.loaders.omama_loader import OmamaLoader
import pydicom as dicom

omama_loader = OmamaLoader()
data = O.Data(data_loader=omama_loader, load_cache=True)


def test_filter_2d_cancer_794x990():
    data.filter_data(_3d=False, row_max=794, col_max=990, labels=["IndexCancer"])
    assert len(data) == int(4574)
    for i in range(len(data)):
        assert data[i].Rows <= 794
        assert data[i].Columns <= 990
    data.reset_data()


def test_filter_2d_cancer_age_between_40_and_60_yrs():
    min_age = 40
    max_age = 60
    known_total = 8335
    data.filter_data(_3d=False, age_range=[min_age, max_age], labels=["IndexCancer"])
    assert len(data) == int(known_total)
    for i in range(len(data)):
        age = int(data[i].PatientAge[:-1])
        assert min_age * 12 <= age <= max_age * 12
    data.reset_data()


def test_total_dicoms_count():
    assert len(data._image_paths) == int(967991)


def test_total_2D_dicoms_count():
    assert data._stats.total_2d_all == int(895834)


def test_3D_dicoms_count():
    assert data._stats.total_3d_all == int(72157)


def test_total_cancer_count():
    assert data._stats.total_cancer == int(15341)


def test_total_preindex_cancer_count():
    assert data._stats.total_preindex == int(1915)


def test_total_noncancer_count():
    assert data._stats.total_noncancer == int(922925)


def test_total_no_label_count():
    assert data._stats.total_no_label == int(27810)


# test 1 from the ast data set
def test_get_study_2_25_91997877614994112660231120982170991143():
    test = data.get_study(
        study_instance_uid="2.25.91997877614994112660231120982170991143",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.91997877614994112660231120982170991143"
        )
    )
    assert dicom_file_names == [
        "BT.2.25.133578829979549276282493899342637312456",
        "BT.2.25.199510797173497179962471794465489715110",
        "BT.2.25.268278220050446465893620855485614331550",
        "BT.2.25.308636611583627994365513503537421622017",
        "DXm.2.25.122357041307285191056766880564163092479",
        "DXm.2.25.127891839803620276931506694446691185443",
        "DXm.2.25.143904890367711998039501020823561123212",
        "DXm.2.25.215233846285930639228802564638377502072",
        "DXm.2.25.30018004646500957330100131098672387777",
        "DXm.2.25.37394219360165889160095494244236789652",
        "DXm.2.25.77366584578923190366573961995292517128",
        "DXm.2.25.83048405509855196194140638839982839402",
    ]


# test 2 from the ast data set
def test_get_study_2_25_326253637286692069639997125359570634434():
    test = data.get_study(
        study_instance_uid="2.25.326253637286692069639997125359570634434",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.326253637286692069639997125359570634434"
        )
    )
    assert dicom_file_names == [
        "DXm.2.25.105978565340159574880325407463054555761",
        "DXm.2.25.117481087661937347732463732266874390794",
        "DXm.2.25.225701712199626620167871073422214931331",
        "DXm.2.25.264165497746919475627869771433530705028",
        "DXm.2.25.26702276959195919846234468635857732385",
        "DXm.2.25.298430633778652753991074494542763799631",
        "DXm.2.25.36792262617208365540569261505917074459",
        "DXm.2.25.91582918310173551335557752923146101123",
    ]


# test 3 from the ast data set
def test_get_study_2_25_51210434826926899234652970476290289005():
    test = data.get_study(
        study_instance_uid="2.25.51210434826926899234652970476290289005",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.51210434826926899234652970476290289005"
        )
    )
    assert dicom_file_names == [
        "DXm.2.25.100352479040876694476459163906354228192",
        "DXm.2.25.110500082796893064992341223591794846863",
        "DXm.2.25.246770720913772109528493221568235723584",
        "DXm.2.25.85839855885371616746789272392413108352",
    ]


# test 1 from the dh0new data set
def test_get_study_2_25_244645250224793309149323750914470862805():
    test = data.get_study(
        study_instance_uid="2.25.244645250224793309149323750914470862805",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.244645250224793309149323750914470862805"
        )
    )
    assert dicom_file_names == [
        "BT.2.25.183099955415281116914712771648830221152",
        "BT.2.25.262865476797393040530676538488706881675",
        "BT.2.25.27531615217925995878853246251945355056",
        "BT.2.25.279857401325399316315982112360734534624",
        "DXm.2.25.208773140614120108093061665484355577003",
        "DXm.2.25.246988060796335538337923196722292426261",
        "DXm.2.25.306288091376045830100499981104600851691",
        "DXm.2.25.81576056370654463210693851855932156432",
    ]


# test 2 from the dh0new data set
def test_get_study_2_25_169889250524454767844791634052759492816():
    test = data.get_study(
        study_instance_uid="2.25.169889250524454767844791634052759492816",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.169889250524454767844791634052759492816"
        )
    )
    assert dicom_file_names == [
        "BT.2.25.178172329289329459861311354587193959897",
        "BT.2.25.21034237040324558805453306117686326601",
        "BT.2.25.269504884488990693746855630723589471639",
        "BT.2.25.41076498352467273057422732452384468622",
        "DXm.2.25.110944609156687826885970390252351995430",
        "DXm.2.25.145570256623616165270350873729118098505",
        "DXm.2.25.166961492959202687938835190168750061774",
        "DXm.2.25.174983401773487899352079740868850755249",
    ]


# test 3 from the dh0new data set
def test_get_study_2_25_164927977049968039928497684194369501756():
    test = data.get_study(
        study_instance_uid="2.25.164927977049968039928497684194369501756",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.164927977049968039928497684194369501756"
        )
    )
    assert dicom_file_names == [
        "BT.2.25.109001031854333001590791420296943887438",
        "BT.2.25.311253887445111265702887149525084615131",
        "BT.2.25.338416072821826283040538903429673943028",
        "BT.2.25.59394539811924965592165335199558649777",
        "DXm.2.25.161770506925715681567243883627977212136",
        "DXm.2.25.215164452061547351076674733580217736798",
        "DXm.2.25.239131946275796764206128619477315164473",
        "DXm.2.25.38776385016120100403246464852642660222",
    ]


# test 1 from the dh2 data set
def test_get_study_2_25_31962288804401111432979100537418851933():
    test = data.get_study(
        study_instance_uid="2.25.31962288804401111432979100537418851933",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.31962288804401111432979100537418851933"
        )
    )
    assert dicom_file_names == [
        "BT.2.25.241784170362777991341372937858753437645",
        "BT.2.25.307899669196959193405490421180141184901",
        "BT.2.25.323858110923411924017608712206116789730",
        "BT.2.25.994409576962329304734380316345779381",
        "DXm.2.25.16034944984939375755303744459645636104",
        "DXm.2.25.191503908200024797892822602341769787854",
        "DXm.2.25.231677586891273920337378825361769595513",
        "DXm.2.25.313281763004765406299968560217484581906",
    ]


# test 2 from the dh2 data set
def test_get_study_2_25_250521486960293924609794685547491809210():
    test = data.get_study(
        study_instance_uid="2.25.250521486960293924609794685547491809210",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.250521486960293924609794685547491809210"
        )
    )
    assert dicom_file_names == [
        "BT.2.25.132551716166627810536070196153219327528",
        "BT.2.25.17988732764240595981178698766096551769",
        "BT.2.25.243626102266375806172949495437038127927",
        "BT.2.25.293242348027059584794679725745192814215",
        "DXm.2.25.111543397832892764655510774859070220978",
        "DXm.2.25.227641055277842358492372143763316513061",
        "DXm.2.25.257456260880969287293973832731449310320",
        "DXm.2.25.47256788809798622804178207403873269851",
    ]


# test 3 from the dh2 data set
def test_get_study_2_25_23697001741413135128882662590024074006():
    test = data.get_study(
        study_instance_uid="2.25.23697001741413135128882662590024074006",
        verbose=True,
        timing=True,
    )
    dicom_file_names = sorted(
        data._files_in_series(
            study_instance_uid="2.25.23697001741413135128882662590024074006"
        )
    )
    assert dicom_file_names == [
        "BT.2.25.235082982634706523979783007555960580576",
        "BT.2.25.37243016938045516324394012516922804369",
        "BT.2.25.46184881038078258419103219076910542642",
        "BT.2.25.56148004588019631584657756243042766049",
        "DXm.2.25.167348012638942493905999335217470855277",
        "DXm.2.25.168482995347963424884184985465834886243",
        "DXm.2.25.264333323522953468851895541438451254233",
        "DXm.2.25.55835190419512490214086525167041602377",
    ]


def test_get_image_1():
    img1 = data.get_image(dicom_name="DXm.2.25.167348012638942493905999335217470855277")
    ds = dicom.dcmread(
        r"/raid/data01/deephealth/dh_dh2/2.25.23697001741413135128882662590024074006/DXm.2.25"
        r".167348012638942493905999335217470855277"
    )
    img = data.get_image(
        image_id=data._image_id(
            dicom_name="DXm.2.25.167348012638942493905999335217470855277"
        )
    )
    shape = ds.pixel_array.shape
    assert img1.shape == shape
    assert img.shape == shape
    assert img1.filePath == img.filePath
    assert img1.label == img.label
    assert img1.pixels.all() == img.pixels.all()


def test_get_image_2():
    img1 = data.get_image(dicom_name="BT.2.25.46184881038078258419103219076910542642")
    ds = dicom.dcmread(
        r"/raid/data01/deephealth/dh_dh2/2.25.23697001741413135128882662590024074006/BT.2.25"
        r".46184881038078258419103219076910542642"
    )
    img = data.get_image(
        image_id=data._image_id(
            dicom_name="BT.2.25.46184881038078258419103219076910542642"
        )
    )
    shape = ds.pixel_array.shape
    assert img1.shape == shape
    assert img.shape == shape
    assert img1.filePath == img.filePath
    assert img1.label == img.label
    assert img1.pixels.all() == img.pixels.all()


def test_get_image_3():
    img1 = data.get_image(dicom_name="DXm.2.25.161770506925715681567243883627977212136")
    ds = dicom.dcmread(
        r"/raid/data01/deephealth/dh_dh0new/2.25.164927977049968039928497684194369501756/DXm.2.25"
        r".161770506925715681567243883627977212136"
    )
    img = data.get_image(
        image_id=data._image_id(
            dicom_name="DXm.2.25.161770506925715681567243883627977212136"
        )
    )
    shape = ds.pixel_array.shape
    assert img1.shape == shape
    assert img.shape == shape
    assert img1.filePath == img.filePath
    assert img1.label == img.label
    assert img1.pixels.all() == img.pixels.all()


def test_get_image_4():
    img1 = data.get_image(dicom_name="DXm.2.25.81576056370654463210693851855932156432")
    ds = dicom.dcmread(
        r"/raid/data01/deephealth/dh_dh0new/2.25.244645250224793309149323750914470862805/DXm.2.25"
        r".81576056370654463210693851855932156432"
    )
    img = data.get_image(
        image_id=data._image_id(
            dicom_name="DXm.2.25.81576056370654463210693851855932156432"
        )
    )
    shape = ds.pixel_array.shape
    assert img1.shape == shape
    assert img.shape == shape
    assert img1.filePath == img.filePath
    assert img1.label == img.label
    assert img1.pixels.all() == img.pixels.all()


def test_get_image_5():
    img1 = data.get_image(dicom_name="DXm.2.25.36792262617208365540569261505917074459")
    ds = dicom.dcmread(
        r"/raid/data01/deephealth/dh_dcm_ast/2.25.326253637286692069639997125359570634434/DXm.2.25"
        r".36792262617208365540569261505917074459"
    )
    img = data.get_image(
        image_id=data._image_id(
            dicom_name="DXm.2.25.36792262617208365540569261505917074459"
        )
    )
    shape = ds.pixel_array.shape
    assert img1.shape == shape
    assert img.shape == shape
    assert img1.filePath == img.filePath
    assert img1.label == img.label
    assert img1.pixels.all() == img.pixels.all()


def test_get_image_6():
    img1 = data.get_image(dicom_name="DXm.2.25.117481087661937347732463732266874390794")
    ds = dicom.dcmread(
        r"/raid/data01/deephealth/dh_dcm_ast/2.25.326253637286692069639997125359570634434/DXm.2.25"
        r".117481087661937347732463732266874390794"
    )
    img = data.get_image(
        image_id=data._image_id(
            dicom_name="DXm.2.25.117481087661937347732463732266874390794"
        )
    )
    shape = ds.pixel_array.shape
    assert img1.shape == shape
    assert img.shape == shape
    assert img1.filePath == img.filePath
    assert img1.label == img.label
    assert img1.pixels.all() == img.pixels.all()


def test_next_image_2d_cancer():
    length = 40
    img = []
    generator = data.next_image(_2d=True, label=OL.Label.CANCER)
    for i in range(length):
        img.append(next(generator))
    assert len(img) == length
    for i in range(length):
        assert img[i].label == OL.Label.CANCER
        assert len(img[i].shape) == 2


def test_next_image_2d_non_cancer():
    length = 40
    img = []
    generator = data.next_image(_2d=True, label=OL.Label.NONCANCER)
    for i in range(length):
        img.append(next(generator))
    assert len(img) == length
    for i in range(length):
        assert img[i].label == OL.Label.NONCANCER
        assert len(img[i].shape) == 2


def test_next_image_2d_preindex_cancer():
    length = 40
    img = []
    generator = data.next_image(_2d=True, label=OL.Label.PREINDEX)
    for i in range(length):
        img.append(next(generator))
    assert len(img) == length
    for i in range(len(img)):
        assert img[i].label == OL.Label.PREINDEX
        assert len(img[i].shape) == 2


def test_next_image_3d_cancer():
    length = 4
    img = []
    generator = data.next_image(_3d=True, label=OL.Label.CANCER)
    for i in range(length):
        img.append(next(generator))
    assert len(img) == length
    for i in range(len(img)):
        assert img[i].label == OL.Label.CANCER
        assert len(img[i].shape) == 3


def test_next_image_3d_non_cancer():
    length = 4
    img = []
    generator = data.next_image(_3d=True, label=OL.Label.NONCANCER)
    for i in range(length):
        img.append(next(generator))
    assert len(img) == length
    for i in range(len(img)):
        assert img[i].label == OL.Label.NONCANCER
        assert len(img[i].shape) == 3


def test_next_image_3d_preindex_cancer():
    generator = data.next_image(_3d=True, label=OL.Label.PREINDEX)
    assert len(list(generator)) == 0


def test_generator_2d_cancer_nonrandom():
    length = 20
    generator1 = data.next_image(_2d=True, label=OL.Label.CANCER)
    generator2 = data.next_image(_2d=True, label=OL.Label.CANCER)
    img1 = []
    img2 = []
    for i in range(length):
        img1.append(next(generator1))
        img2.append(next(generator2))
    assert len(img1) == length
    assert len(img2) == length
    for i in range(length):
        assert img1[i].label == img2[i].label
        assert img1[i].filePath == img2[i].filePath
        assert len(img1[i].shape) == 2
        assert len(img2[i].shape) == 2


def test_generator_2d_cancer_random():
    length = 20
    generator1 = data.next_image(_2d=True, label=OL.Label.CANCER, randomize=True)
    generator2 = data.next_image(_2d=True, label=OL.Label.CANCER, randomize=True)
    img1 = []
    img2 = []
    for i in range(length):
        img1.append(next(generator1))
        img2.append(next(generator2))
    assert len(img1) == length
    assert len(img2) == length
    for i in range(length):
        assert img1[i].label == img2[i].label
        assert img1[i].filePath != img2[i].filePath
        assert len(img1[i].shape) == 2
        assert len(img2[i].shape) == 2


def test_generator_3d_cancer_nonrandom():
    length = 3
    generator1 = data.next_image(_3d=True, label=OL.Label.CANCER)
    generator2 = data.next_image(_3d=True, label=OL.Label.CANCER)
    img1 = []
    img2 = []
    for i in range(length):
        img1.append(next(generator1))
        img2.append(next(generator2))
    assert len(img1) == length
    assert len(img2) == length
    for i in range(length):
        assert img1[i].label == img2[i].label
        assert img1[i].filePath == img2[i].filePath
        assert len(img1[i].shape) == 3
        assert len(img2[i].shape) == 3


def test_generator_3d_cancer_random():
    length = 3
    generator1 = data.next_image(_3d=True, label=OL.Label.CANCER, randomize=True)
    generator2 = data.next_image(_3d=True, label=OL.Label.CANCER, randomize=True)
    img1 = []
    img2 = []
    for i in range(length):
        img1.append(next(generator1))
        img2.append(next(generator2))
    assert len(img1) == length
    assert len(img2) == length
    for i in range(length):
        assert img1[i].label == img2[i].label
        assert img1[i].filePath != img2[i].filePath
        assert len(img1[i].shape) == 3
        assert len(img2[i].shape) == 3


def test_generator_all_unique_per_generator_nonrandom():
    length = 30
    generator1 = data.next_image(_2d=True)
    img_dict = {}
    for i in range(length):
        img = next(generator1)
        if img_dict.get(img.filePath) is not None:
            assert False
        else:
            img_dict[img.filePath] = i
    assert len(img_dict) == length


def test_generator_all_unique_per_generator_random():
    length = 30
    generator1 = data.next_image(_2d=True, randomize=True)
    img_dict = {}
    for i in range(length):
        img = next(generator1)
        if img_dict.get(img.filePath) is not None:
            assert False
        else:
            img_dict[img.filePath] = i
    assert len(img_dict) == length
