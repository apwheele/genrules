'''
Genetic algorithm to conduct 
association rules, optimizing
for a binary outcome (relative risk)
'''

import pandas as pd
import numpy as np
import copy
import random
from scipy.stats import norm
from evol import Population, Evolution
import itertools
import networkx as nx

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib.plt as plt


def feat_map(x):
    fm = {}
    for v in list(x):
        # Get not missing
        vnm = pd.unique(x.loc[x[v].notna(), v]).tolist()
        fm[v] = vnm
    return fm

def sel_rand(fm):
    res_sel = []
    for k,i in fm.items():
        res_sel += [np.random.choice(i)]
    return res_sel

def sel_any(fm, prob):
    sel_var = sel_rand(fm)
    var_zer = np.random.binomial(1,prob,len(sel_var))
    res_li = []
    for s,v in zip(sel_var,var_zer):
        if v == 0:
            res_li.append(np.nan)
        else:
            res_li.append(s)
    return res_li

class genrules():
    def __init__(self,data,y_var,x_vars,w_var,k=3,penrat=32,
                 clip_val=1e-12,min_samp=20,mut_prob=0.5,leader_tot=100):
        """
        generating initial object and attaching data
        """
        self.y_var = y_var
        self.x_vars = x_vars
        self.w_var = w_var
        self.x = data[x_vars]
        self.y = data[y_var]
        self.w = data[w_var]
        self.len_x = len(x_vars)
        self.min_samp = min_samp
        self.penrat = penrat
        # Creating the offset ratio for the penalty term
        topy = np.log2(penrat)
        boty =  1 #np.log2(2)
        ratp = (topy - boty)/(0.5 - 1)
        const = topy + (0.5*ratp)
        self.top_clip = topy
        self.penterms = [ratp,const]
        # Creating feature map for x variables
        self.fm = feat_map(data[x_vars])
        self.mut_prob = mut_prob
        self.clip_val = clip_val
        # Creating the initial population
        res = []
        for i,v in enumerate(self.x_vars):
            for c in self.fm[v]:
                base = [None]*self.len_x
                base[i] = c
                res.append(tuple(base.copy()))
        # For larger values of k
        if k > 1:
            sv = set(x_vars)
            for i in range(2,k+1):
                for c in itertools.combinations(x_vars,i):
                    left_over = list(sv - set(c))
                    all_res = self.x.drop_duplicates(subset=c).copy()
                    all_res.loc[:,left_over] = None
                    for r in all_res.itertuples():
                        res.append(r[1:])
        self.init_pop = res
        # Maybe add in random pairs/triples/etc.
        # placeholder for pop Object
        self.evo_pop = None
        self.mutype = 'add'
        # Adding in the leaderboard
        self.leaderboard = None
        self.leaderboard_tot = leader_tot
        self.generation = 0
    def opt_stats(self, xp):
        """
        function to optimize given data
        """
        # calculate cross-tab for each category
        npx = np.array(xp)
        xvsel = (npx != None)
        # If all zeroes, return 0s across board
        if xvsel.sum() == 0:
            return (0,0,0,0,0)
        # Else go through the motions
        xdsel = 1*((self.x.iloc[:,xvsel] == npx[xvsel]).all(axis=1))
        # If not enough cases, return 0s
        if (xdsel*self.w).sum() < self.min_samp:
            return (0,0,0,0,0)
        ct = pd.crosstab(index=xdsel,columns=self.y,values=self.w,aggfunc=sum).fillna(self.clip_val)
        # If shape is not right (0 values in some row/column, return all 0s)
        if (ct.shape[0] != 2) | (ct.shape[1] != 2):
            return (0,0,0,0,0)
        nt = np.array(ct)
        # calculate relative risk (clipping so no exact zeroes)
        selp = (nt[0,1]/(nt[0,0] + nt[0,1])) #0 in numerator is ok
        obsp = (nt[1,1]/(nt[1,0] + nt[1,1])).clip(self.clip_val)
        relrisk = selp/obsp
        # calculate p-value for log relative risk
        log_rr = np.log(relrisk.clip(self.clip_val))
        l1 = nt[0,0]/(nt[0,1]*(nt[0,1] + nt[0,0]))
        l2 = nt[1,0]/(nt[1,1]*(nt[1,1] + nt[1,0]))
        se_rr = np.sqrt( (l1 + l2) )
        z = log_rr/se_rr
        p = 1 - norm.cdf(z) # only positive part
        # The penalty term
        fitness = np.log2(relrisk).clip(-self.top_clip,self.top_clip) + p*self.penterms[0] + self.penterms[1]
        # The relative risk is penalized by p-value
        return [relrisk, log_rr, se_rr, p, fitness]
    def opt_func(self, xp):
        #print(xp) #useful for debugging
        return self.opt_stats(xp)[-1]
    def mut_rand(self, x):
        # x should be a list of the current values
        # only mutate single variable if over probability
        # should also consider mutations back to None
        if np.random.uniform() > self.mut_prob:
            return x
        # if muttype is remove
        elif self.mutype == 'remove':
            res = []
            for v in x:
                if np.random.uniform() < self.mut_prob:
                    res.append(None)
                else:
                    res.append(v)
            return tuple(res)
        # else add in single new type
        else:
            xn = list(x)
            # pick a random variable from the list
            rv = np.random.choice(self.len_x)
            # pick a new variant
            rvs = self.x_vars[rv]
            left_overs = set(self.fm[rvs]) - set([x[rv]])
            # Add back in
            xn[rv] = np.random.choice(list(left_overs))
            return tuple(xn)
    def child(self, x1, x2):
        # making children given two input lists
        p = np.random.uniform(size=self.len_x).tolist()
        xnew = []
        for x1,x2,p in zip(x1,x2,p):
            if (x1 is None) & (x2 is None):
                xnew.append(None)
            elif (x1 is None) & (x2 is not None):
                xnew.append(x2)
            elif (x1 is not None) & (x2 is None):
                xnew.append(x1)
            else:
                if p < 0.5:
                    xnew.append(x1)
                else:
                    xnew.append(x2)
        # random mutation
        #xmut = self.mut_rand(xnew)
        return tuple(xnew)
    def pick_parents(self, pop):
        return random.choices(pop, k=2)
    def table_pop(self):
        r1 = []
        if self.leaderboard is None:
            for i in self.init_pop:
                cr = list(i)
                vals = list(self.opt_stats(cr))
                lab = [{v:l for v,l in zip(self.x_vars,cr) if l is not None}]
                r1.append(cr + vals + lab)
        else:
            for i in self.evo_pop:
                cr = list(i.chromosome)
                vals = list(self.opt_stats(cr))
                lab = [{v:l for v,l in zip(self.x_vars,cr) if l is not None}]
                r1.append(cr + vals + lab)
        rdat = pd.DataFrame(r1, columns=self.x_vars + ['relrisk','log_rr','se_rr','pval','fitness'] + ['label'])
        rdat.drop_duplicates(subset=self.x_vars,inplace=True)
        rdat.sort_values(by='fitness',ascending=False,inplace=True,ignore_index=True)
        rdat = rdat[rdat['fitness'] > 0].reset_index(drop=True)
        if self.leaderboard is None:
            self.leaderboard = rdat
            print(f"Initial candidates added to leaderboard {rdat.shape[0]}")
        else:
            old = self.leaderboard.copy()
            old['new'] = 0
            rdat['new'] = 1
            combined = pd.concat([old,rdat],axis=0)
            combined.drop_duplicates(subset=self.x_vars,keep='first',inplace=True)
            combined.sort_values(by='fitness',ascending=False,inplace=True,ignore_index=True)
            if combined.shape[0] > self.leaderboard_tot:
                combined = combined.head(self.leaderboard_tot).copy()
            print(f"Total new cases added to leaderboard {combined['new'].sum()}")
            self.leaderboard = combined[list(self.leaderboard)].copy()
        return rdat
    def evolve(self, surv_n, rep, set_mute='add', redo_pop=False):
        if self.leaderboard is None:
            print('Creating initial leaderboard')
            tab = self.table_pop()
        #if pop none create it, else evolve that result even further
        if self.evo_pop is None:
            self.evo_pop = Population(chromosomes=self.init_pop,eval_function=self.opt_func,
                           maximize=True)
        elif redo_pop:
            leader_pop = [i[1:] for i in self.leaderboard[ge.x_vars].itertuples()]
            self.evo_pop = Population(chromosomes=leader_pop,eval_function=self.opt_func,
                           maximize=True)
        evo = (Evolution()
               .survive(n=surv_n)
               .breed(parent_picker=self.pick_parents,combiner=self.child)
               .mutate(mutate_function=self.mut_rand))
        for _ in range(rep):
            self.generation += 1
            print(f'\nGeneration {self.generation}')
            self.evo_pop = self.evo_pop.evolve(evo, n=1)
            tab = self.table_pop()
    def active_table(self,type='vars'):
        if type == 'vars':
            res = (~self.leaderboard[self.x_vars].isna()).sum(axis=0)
            res = res.reset_index()
            res.columns = ['Variable','TotActive']
            res.sort_values(by='TotActive',ascending=False,inplace=True,ignore_index=True)
            return res
        else:
            res_list = []
            for l in self.leaderboard['label']:
                for v,a in l.items():
                    res_list.append( (v,a) )
            res_data = pd.DataFrame(res_list,columns=['Variable','Attribute'])
            res_data['TotActive'] = 1
            res_grp = res_data.groupby(['Variable','Attribute'],as_index=False).sum()
            res_grp.sort_values(by=['TotActive','Variable'],ascending=False,inplace=True,ignore_index=True)
            return res_grp
    def network(self,type='att'):
        labs = self.leaderboard['label'].tolist()
        edges = []
        if type == 'att':
            for l in labs:
                v = [v + ":" + str(k) for v,k in l.items()]
                for c in itertools.combinations(v,2):
                    edges.append(c)
        else:
            for l in labs:
                v = [v for v,k in l.items()]
                for c in itertools.combinations(v,2):
                    if c[0] < c[1]:
                        edges.append(c)
                    else:
                        edges.append((c[1],c[0]))
        # Dataframe and agg weights
        edge_dat = pd.DataFrame(edges, columns=['source','target'])
        edge_dat['weight'] = 1
        edge_dat = edge_dat.groupby(['source','target'], as_index=False).sum()
        # Now creating networkx graph
        g = nx.from_pandas_edgelist(edge_dat,edge_attr='weight')
        # Now making nice network graph
        return g

# ToDo
# Add in TotN to tables
# CoMorbid function
# Risk Difference Black/White (totally new class)



########################
## Making a nice network graph
#et = ge.network()
#weights = nx.get_edge_attributes(et,'weight').values()
#nx.draw(et, width=list(weights), with_labels=True)
#plt.show()
########################