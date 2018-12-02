import os
from operator import add
from PySpark import SparkContext

def getParentsChildren(rdd):
    return rdd.flatMap(
        lambda x: [x,x[::-1]]
    ).mapValues(
        lambda x: [x]
    ).reduceByKey(add).map(
        lambda x: (x[0],{'p': [z for z in x[1] if z<x[0]],
                         'c': [z for z in x[1] if z>x[0]]})
    )

def BigStar(pC):
    l = len(pC[1]['p'])
    if l:
        min_parent = min(pC[1]['p'])
        new_edges = [[z,min_parent] for z in pC[1]['c']]
        if l>1:
            new_edges.append([-1,-1])
    else:
        new_edges = [[z,pC[0]] for z in pC[1]['c']]
    return new_edges

def SmallStar(pC):
    if pC[1]['p']:
        min_parent = min(pC[1]['p'])
        new_edges = [[z, min_parent] for z in pC[1]['p'] + [pC[0]]]
        return new_edges
    else:
        return [[-1,-1]] if pC[0]==-1 else []

def iteration(edges):
    return getParentsChildren(
        getParentsChildren(edges).flatMap(BigStar)
    ).flatMap(SmallStar)

def ConnectedComponents(edges):
    rdds = [edges]
    for i in range(10**7):
        print(i)
        rdds.append(iteration(rdds[-1]))
        rdds[-1].cache()
        if len(rdds[-1].lookup(-1)) == 0:
            break
        rdds[-2].unpersist()
    return rdds[-1].reduceByKey(lambda x,y: x)

sc = SparkContext(appName='A2_instacart')
edges = sc.textFile(
	'/user/vliunda/data/edges.txt'
	).map(
		lambda x: [int(y) for y in x.split(' ')[1:]]
	)

res = ConnectedComponents(edges)
lines = res.map(lambda edge: ' '.join(str(v) for v in edge))
lines.saveAsTextFile('./res.txt')