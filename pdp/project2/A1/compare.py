files_actual = [f'{i}_actual.csv' for i in range(4)]
files_mpi = [f'{i}_mpi.csv' for i in range(4)]

def readfile(filename):
    with open(filename,'r') as f:
        ints = [int(x) for x in f.read().split(',')[:-1]]
    return ints

act = []
mpi = []

for f in files_actual:
    act += readfile(f)

for f in files_mpi:
    mpi += readfile(f)

print(sorted(act) == sorted(mpi))