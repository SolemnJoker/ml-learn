import tree as t
import treePlotter as tp
import os

f = open(os.path.dirname(__file__) +'/lenses.txt')
lenses = [r.strip().split('\t') for r in f.readlines()]
lensesLabel = ['age','prescript','astigmatic','tearRate']
lensesTree = t.createTree(lenses,lensesLabel)
tp.createPlot(lensesTree)
fmt = '%10s'
print [fmt % x for x in lensesLabel]
for lense in lenses:
    print [fmt % x for x in lense],t.classify(lensesTree,lensesLabel,lense[0:-1])