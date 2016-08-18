import numpy
import h5py

f = open("auto-mpg.data","r")
D = []
for line in f:
    line = line.strip()
    try:
        data = line.split()
        data = [float(d) for d in data[:8]]
        D.append(data)
    except:
        pass
D = numpy.asarray(D,dtype='float')
with h5py.File('auto-mpg.hdf5', 'w') as F:
    F.create_dataset('dataset', data=D)
