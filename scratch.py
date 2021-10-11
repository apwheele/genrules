'''
Scratch example
run from root of the project
Andy Wheeler
'''

import pandas as pd
from src import genrules
from src import dataprep


###################################################
# EXAMPLE DATA ANALYSIS FOR infection

infect = 'CBAB01' #infection

# Set of demographic variables
rhv = dataprep.demo

# Do some data analysis to prep
inf_dat = dataprep.prep_dat(infect,rhv)

# Set up object with all defaults
ge = genrules.genrules(data=inf_dat,y_var=infect,x_vars=rhv)

# Evolve the pop 5 generations, see what additional rules we discover
ge.evolve(rep=5)

# We can check out the top rules in the current leaderboard
tb = ge.leaderboard
tb[['relrisk','pval','tot_n','out_n','label']].head(20)

# row 12, CRace3 hispanic, poverty 3 < 100% poverty level, morbidly obese, 5/28, RR3
# row 19, CRace2 black, other insurance, 5/30 RR 2.8

# Can check the commonly activated attributes among the variables
ge.active_table(type='att')

# One can then pull out Ins_Govt to check out additional
# co-morbidities
tb.loc[tb['ins_type'] == 'Ins_Govt',rhv + ['relrisk']]
#####################################################

###################################################
# EXAMPLE DATA ANALYSIS FOR post-partum depression

dep = 'CMAE04a1c'

rhv = dataprep.demo

# Do some data analysis to prep
dep_dat = dataprep.prep_dat(dep,rhv)

# Check all triplets of characteristics
dep_ge = genrules.genrules(data=dep_dat,y_var=dep,x_vars=rhv,k=3)

# Do not evolve, just see up to triples
dep_ge.evolve(rep=0)

tb = dep_ge.leaderboard
tb[['relrisk','pval','tot_n','out_n','label']].head(10)
###################################################

###################################################
# EXAMPLE DATA ANALYSIS FOR hypertensive/clampsia

hyper = 'CBAC01'
# These are dummy variables representing 
# Drugs taken in first 2 months of pregnancy
rhv = dataprep.drug_dummyvars
hyper_dat = dataprep.prep_dat(hyper,rhv)

# Only do single to start, so generations are faster
# Lessen penalty for extra variables and smaller samples
hy_ge = genrules.genrules(data=hyper_dat,y_var=hyper,x_vars=rhv,k=1,pen_var=0,min_samp=30)

hy_ge.evolve(rep=6)
# set_mute sets mutations to remove attributes
hy_ge.evolve(rep=6,set_mute='remove')

tb = hy_ge.leaderboard
tb[['relrisk','pval','tot_n','out_n','label']].head(20)

drug_act = hy_ge.active_table(type='att')
drug_act[drug_act['Attribute'] == 1]

# 610 is Influenza
# 500's are vitamins
# 240 Hormonal Contraceptive
# 103 Triptans
# 230 Steroids

# 112 Nitrofurantoin 
# 341 Insulin 

# Lets look at Insulin
tb.loc[tb['Drug_341'] == 1,['relrisk','pval','tot_n','out_n','label']]
###################################################