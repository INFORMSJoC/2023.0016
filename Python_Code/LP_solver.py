from gurobipy import *
import time
import numpy as np

# Class for compute LP and retrieve LP results in clusters
class LP_Solution():
    def __init__(self, num_views, num_targets, cost_dict):
        self.n = num_views
        self.q = num_targets
        self.cost_dict = cost_dict

    # Function to solve the LP problem
    def solver(self):
        m = Model()
        var_dic = {}

        obj = 0
        for t in range(self.n):
            for h in range(t+1, self.n):
                for i in range(self.q):
                    for j in range(self.q):
                        s = "x"+str(t)+","+str(h)+","+str(i)+","+str(j)
                        # var_dic[(t,h,i,j)] = m.addVar(name = s, lb=0, ub=1)
                        var_dic[(t, h, i, j)] = m.addVar(name=s, vtype=GRB.BINARY)
                        cost = self.cost_dict[(t+1, h+1)][i][j]
                        obj = obj + var_dic[(t, h, i, j)] * cost

        m.setObjective(obj, GRB.MINIMIZE)
        for t in range(self.n):
            for h in range(t+1, self.n):
                for i in range(self.q):
                    tmp = 0
                    for j in range(self.q):
                        tmp += var_dic[(t, h, i, j)]
                    m.addConstr(tmp == 1, "c1")

        for t in range(self.n):
            for h in range(t + 1, self.n):
                for j in range(self.q):
                    tmp = 0
                    for i in range(self.q):
                        tmp += var_dic[(t,h,i,j)]
                    m.addConstr(tmp == 1, "c2")

        for t in range(self.n-1):
            for h in range(t+1, self.n):
                for l in range(h+1,self.n):
                    for i in range(self.q):
                        for j in range(self.q):
                            for k in range(self.q):
                                m.addConstr(
                                    var_dic[(t, h, i, j)] + var_dic[(h, l, j, k)] - var_dic[(t, l, i, k)] - 1 <= 0, "c3")
                                m.addConstr(
                                    var_dic[(h, l, j, k)] + var_dic[(t, l, i, k)] - var_dic[(t, h, i, j)] - 1 <= 0, "c4")
                                m.addConstr(
                                    var_dic[(t, l, i, k)] + var_dic[(t, h, i, j)] - var_dic[(h, l, j, k)] - 1 <= 0, "c5")

        m.Params.LogToConsole = 0
        m.optimize()

        return m


    # Function to obtain the clustering results from the LP results
    def getPath(self, m):
        cluster = np.zeros((self.q, self.n),dtype=int)
        for i in range(self.q):
            cluster[i][0] = i

        for t in range(self.n):
            for i in range(self.q):
                for j in range(self.q):
                    s = "x" + str(t) + "," + str(t + 1) + "," + str(i) + "," + str(j)
                    for v in m.getVars():
                        if v.varName == s and round(v.x)==1:
                            cluster[j][t+1] = cluster[i][t]
        return cluster




