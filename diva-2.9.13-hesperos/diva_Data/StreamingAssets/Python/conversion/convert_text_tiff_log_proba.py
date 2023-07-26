from pandas import read_csv
from tifffile import imwrite
import math

data = read_csv('log_proba_state_test.csv', header=None)

label_0 = data[0]
label_1 = data[1]

for i in range(len(label_0)):
    label_0[i] = 255 * math.exp(label_0[i])
    label_1[i] = 255 * math.exp(label_1[i])

label_0 = label_0.astype('uint8')
label_0 = label_0.values.reshape((64,64,64))

label_1 = label_1.astype('uint8')
label_1 = label_1.values.reshape((64,64,64))

# print(label_1)

imwrite('proba_0.tif', label_0, photometric='minisblack')
imwrite('proba_1.tif', label_1, photometric='minisblack')

