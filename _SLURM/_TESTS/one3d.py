import sys

sys.path.insert(0, "../..")
import omama as O

imgs = O.DataHelper.get3D(N=1, cancer=True, randomize=True, timing=True)
pred = O.DeepSight.run(imgs, timing=True)
print(pred)
