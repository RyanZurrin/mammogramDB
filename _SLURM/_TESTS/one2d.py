import sys, os

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../")
import omama as O

imgs = O.DataHelper.get2D(N=1, cancer=True, randomize=True, timing=True)
pred = O.DeepSight.run(imgs, timing=True)
print(pred)
