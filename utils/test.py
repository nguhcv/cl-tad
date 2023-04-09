import numpy as np


a = np.ones((3,3))
a[0] = np.asarray([1,2,3])
a[2] = np.asarray([8,9,10])

a[1] = np.asarray([3,4,5])
print(a)


b = np.asarray([2,2,2])

a[:] = a[:] - b

print(a)