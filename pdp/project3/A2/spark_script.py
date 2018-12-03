from pyspark import SparkContext
from operator import add
from sys import argv 
def reverse(x):
    yield x
    yield x[::-1]

def LargeStar(pC):
    l = len(pC[1][0])
    if l:
        min_parent = min(pC[1][0])
        new_edges = [(z, min_parent) for z in pC[1][1]]
        if l>1:
            new_edges.append((-1,-1))
    else:
        new_edges = [(z, pC[0]) for z in pC[1][1]]
    return new_edges

def SmallStar(vP):
    if len(vP[1]):
        min_parent = min(vP[1])
        new_edges = [(z, min_parent) for z in vP[1]]
        new_edges.append((vP[0], min_parent))
        return new_edges
    else:
        return [(-1,-1)] if vP[0] == -1 else ()

def iteration(edges):
    all_pairs = edges.flatMap(reverse)

    parents = all_pairs.filter(lambda x: x[0]>x[1])
    children = all_pairs.filter(lambda x: x[0]<x[1])

    return parents.groupWith(
            children
        ).flatMap(
            LargeStar
        ).flatMap(
            reverse
        ).filter(
            lambda x: (x[0] == -1) | (x[0]>x[1])
        ).groupByKey().flatMap(
            SmallStar
        ).filter(
            lambda x: (x[0] == -1) | (x[0]!=x[1])
        ).distinct()

def ConnectedComponents(edges):
    rdds = [edges]
    for i in range(10**7):
        print(f"\n\niteration: {i}\n\n")
        rdds.append(iteration(rdds[-1]))
        rdds[-1].cache()
        if len(rdds[-1].lookup(-1)) == 0:
            break
        rdds[-2].unpersist()
    return rdds[-1].reduceByKey(lambda x,y: x)

sc = SparkContext(appName='A2')

lines = sc.textFile(argv[1])
edges = lines.map(lambda x: tuple([int(y) for y in x.split(' ')]))

res = ConnectedComponents(edges)

with open('/user/vliunda/data/res.txt','w') as f:
    [f.write(f"{l[0] l[1]}") for l in res.collect()]
