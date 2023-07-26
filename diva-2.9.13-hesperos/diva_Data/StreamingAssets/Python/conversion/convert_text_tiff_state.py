from pandas import read_csv
from tifffile import imwrite

data = read_csv('state_test.csv', header=None)
data = data.astype('uint8')
data = data.values.reshape((64,64,64))

print(data)

imwrite('states.tif', data, photometric='minisblack')

