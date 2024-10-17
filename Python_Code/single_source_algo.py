from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import data_readin
from scipy.optimize import linear_sum_assignment
import LP_solver as solver

# Function to perfrom hungarian algorithm for the 2D assignment problem 
# dist_matrix: input distance matrix
# num_targets: number of targets
# x_clean: solution for the 2D assignment problem
# total_cost: total cost for the 2D assignment problem solution
def assignment_problem(dist_matrix, num_targets):
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    total_cost = 0
    x_clean = np.zeros((num_targets, num_targets))
    for i in range(num_targets):
        x_clean[i, col_ind[i]] = 1
        total_cost = total_cost + dist_matrix[i][col_ind[i]]
    return x_clean, total_cost

# Function for the Forward Heuristic Approach (FHA)
# raw_data_dict: pair-wise distance dictionary for inputs
# num_targets: number of targets
# num_views: number of sensors/stages
# target_clusters: oupput target cluster
# curr_cost: total cost for the oupput target cluster
def forward_process(raw_data_dict, num_targets, num_views):
    head_stage = 1
    start_stage = 2
    prev_stage = 0
    curr_cost = np.zeros((num_targets))
    target_clusters = np.zeros((num_targets, num_views))
    
    for i in range(num_targets):
        target_clusters[i,0] = i

    curr_indicator = np.zeros((num_targets))
    for stage in range(start_stage, num_views+1):
        if stage == 2:
            dist_matrix = raw_data_dict[(head_stage,stage)]
        else:
            dist_matrix = []
            for i in range(num_targets):
                dists = []
                for j in range(num_targets):
                    dist = curr_cost[i]
                    for k in range(head_stage, stage):
                        index = np.where(target_clusters[:, k-1] == i)
                        curr_dist_dict = raw_data_dict[(k, stage)]
                        row_index = int(index[0])
                        dist += curr_dist_dict[row_index][j]
                    dists.append(dist)
                dist_matrix.append(dists)

        ap_sol, total_cost = assignment_problem(dist_matrix, num_targets)
        
        for i in range(num_targets):
            for j in range(num_targets):
                if ap_sol[i,j] == 1:
                    curr_cost[i] = dist_matrix[i][j]
                    target_clusters[j, stage-1] = i
    return target_clusters, curr_cost

# Function for the Multi-Stage/Sensor Recursive Algorithm (MSRA)
# raw_data_dict: pair-wise distance dictionary for inputs
# num_targets: number of targets
# num_views: number of sensors/stages
# target_clusters: oupput target cluster
# curr_cost: total cost for the oupput target cluster
def MSRA(raw_data_dict, num_targets, num_views):
    target_clusters = np.zeros((num_targets, num_views))
    curr_cost = np.zeros((num_targets))
    opt_cost = 99999
    for i in range(num_views-1):
        dist_matrix = raw_data_dict[(i+1, i+2)]
        ap_sol, total_cost = assignment_problem(dist_matrix, num_targets)
        if opt_cost > total_cost:
            opt_sol = ap_sol
            opt_cost = total_cost
            opt_dist_matrix = dist_matrix
            opt_head = i+1
            opt_tail = i+2

    for i in range(num_targets):
        target_clusters[i, opt_head-1] = i

    for i in range(num_targets):
        for j in range(num_targets):
            if opt_sol[i,j] == 1:
                curr_cost[i] = opt_dist_matrix[i][j]
                target_clusters[j, opt_head] = i

    if opt_head != 1:
        for stage in range(opt_head-1, 0, -1):
            dist_matrix = []
            for i in range(num_targets):
                dists = []
                for j in range(num_targets):
                    for k in range(stage, opt_tail):
                        if k == stage:
                            clus = target_clusters[:, k][j]
                            dist = curr_cost[int(clus)]
                        index = np.where(target_clusters[:, k] == clus)
                        col_index = int(index[0])
                        curr_dist_dict = raw_data_dict[(stage, k+1)]
                        dist += curr_dist_dict[i][col_index]
                    dists.append(dist)
                dist_matrix.append(dists)

            ap_sol, total_cost = assignment_problem(dist_matrix, num_targets)

            for i in range(num_targets):
                for j in range(num_targets):
                    if ap_sol[i, j] == 1:
                        curr_cost[j] = dist_matrix[i][j]
                        target_clusters[i, stage - 1] = target_clusters[:, stage][j]

        indicator_map = {}
        for i in range(num_targets):
            indicator_map[target_clusters[i,0]] = i

        for i in range(num_targets):
            for j in range(opt_tail):
                target_clusters[i,j] = indicator_map[target_clusters[i,j]] 

    if opt_tail != num_views:
        for stage in range(opt_tail+1, num_views+1):
            dist_matrix = []
            for i in range(num_targets):
                dists = []
                for j in range(num_targets):
                    dist = curr_cost[i]
                    for k in range(1, stage):
                        index = np.where(target_clusters[:, k-1] == i)
                        curr_dist_dict = raw_data_dict[(k, stage)]
                        row_index = int(index[0])
                        dist += curr_dist_dict[row_index][j]
                    dists.append(dist)
                dist_matrix.append(dists)

            ap_sol, total_cost = assignment_problem(dist_matrix, num_targets)

            for i in range(num_targets):
                for j in range(num_targets):
                    if ap_sol[i,j] == 1:
                        curr_cost[i] = dist_matrix[i][j]
                        target_clusters[j, stage-1] = i
    return target_clusters, curr_cost

# Function for path-based Multi-Stage/Sensor Recursive Algorithm (RMSRA)
# raw_data_dict: pair-wise distance dictionary for inputs
# num_targets: number of targets
# num_views: number of sensors/stages
# target_clusters: oupput target cluster
def MSRA_p(raw_data_dict, num_targets, num_views):
    target_clusters = np.zeros((num_targets, num_views))
    forward_target_clusters = np.zeros((num_targets, num_views))
    backward_target_clusters = np.zeros((num_targets, num_views))
    curr_cost = np.zeros((num_targets))
    re_curr_cost = np.zeros((num_targets))
    all_stages = np.array(range(num_views))
    unfinished_index = list(range(1,num_views+1))
    finished_index = []
    curr_indi = np.zeros((2))
    opt_cost = 99999

    for i in range(1, num_views):
        dist_matrix = raw_data_dict[(i, i+1)]
        ap_sol, total_cost = assignment_problem(dist_matrix, num_targets)
        if opt_cost > total_cost:
            opt_sol = ap_sol
            opt_cost = total_cost
            opt_dist_matrix = dist_matrix
            opt_i = i-1
            opt_j = i

    curr_indi[0] = opt_i
    curr_indi[1] = opt_j
    
    for i in range(num_targets):
        target_clusters[i, opt_i] = i

    for i in range(num_targets):
        for j in range(num_targets):
            if opt_sol[i,j] == 1:
                curr_cost[i] = opt_dist_matrix[i][j]
                target_clusters[j, opt_j] = i

    former_token = False
    after_token = False
    while (curr_indi[0] >= 0 or curr_indi[1] < num_views):
        if curr_indi[0] >= 0:
            former_token = True
            former_indicator = curr_indi[0]
            former_matrix = raw_data_dict[(former_indicator+1, former_indicator+2)]
            former_sol, former_cost = assignment_problem(former_matrix, num_targets)
             
        if curr_indi[1] <= num_views-1:
            after_token = True
            after_indicator = curr_indi[1] + 1
            after_matrix = raw_data_dict[(after_indicator-1, after_indicator)]
            after_sol, after_cost = assignment_problem(after_matrix, num_targets)
            
        if former_token and after_token:
            if former_cost < after_cost:
                opt_sol = former_sol
                curr_stage = curr_indi[0]
                curr_indi[0] -= 1
            else:
                opt_sol = after_sol
                curr_stage = curr_indi[1]
                curr_indi[1] += 1 
            former_token = False
            after_token = False
        elif former_token:
            opt_sol = former_sol
            curr_stage = curr_indi[0]
            curr_indi[0] -= 1
            former_token = False
        elif after_token:
            opt_sol = after_sol
            curr_stage = curr_indi[1]
            curr_indi[1] += 1
            after_token = False
        
        for i in range(num_targets):
            for j in range(num_targets):
                if opt_sol[i,j] == 1:
                    target_clusters[j, int(curr_stage)] = i
                    
    return target_clusters

# Function for path-based Revised Multi-Stage/Sensor Recursive Algorithm (RMSRA)
# raw_data_dict: pair-wise distance dictionary for inputs
# num_targets: number of targets
# num_views: number of sensors/stages
# target_clusters: output target cluster
def RMSRA(raw_data_dict, num_targets, num_views):    
    target_clusters = np.zeros((num_targets, num_views))
    forward_target_clusters = np.zeros((num_targets, num_views))
    backward_target_clusters = np.zeros((num_targets, num_views))
    curr_cost = np.zeros((num_targets))
    re_curr_cost = np.zeros((num_targets))
    all_stages = np.array(range(num_views))
    unfinished_index = list(range(1,num_views+1))
    finished_index = []
    opt_cost = 99999
    for i in range(1, num_views):
        for j in range(i+1, num_views+1):
            dist_matrix = raw_data_dict[(i, j)]
            ap_sol, total_cost = assignment_problem(dist_matrix, num_targets)
            if opt_cost > total_cost:
                opt_sol = ap_sol
                opt_cost = total_cost
                opt_dist_matrix = dist_matrix
                opt_i = i-1
                opt_j = j-1

    finished_index.append(opt_i+1)  
    finished_index.append(opt_j+1)  

    unfinished_index.remove(opt_i+1)
    unfinished_index.remove(opt_j+1)

    for i in range(num_targets):
        target_clusters[i, opt_i] = i

    for i in range(num_targets):
        for j in range(num_targets):
            if opt_sol[i,j] == 1:
                curr_cost[i] = opt_dist_matrix[i][j]
                target_clusters[j, opt_j] = i

    while len(unfinished_index) > 0:
        stage_costs = {}
        stage_sum_costs = {}
        stage_clusters = {}
        for stage in unfinished_index:
            dists_matrix = []
            for i in range(num_targets):
                curr_stage = stage
                dists = []
                for j in range(num_targets):
                    dist = curr_cost[i]
                    for fin_stage in finished_index:
                        clu_index = np.where(target_clusters[:, fin_stage-1] == i)
                        curr_dist_dict = raw_data_dict[(curr_stage, fin_stage)]
                        col_index = int(clu_index[0])
                        dist += curr_dist_dict[j][col_index]
                    dists.append(dist)
                dists_matrix.append(dists)

            ap_sol, total_cost = assignment_problem(dists_matrix, num_targets)

            stage_cluster = np.zeros((num_targets, num_views))
            for i in range(num_targets):
                for j in range(num_views):
                    stage_cluster[i,j] = target_clusters[i,j]

            stage_cost = np.zeros((num_targets))
            for i in range(num_targets):
                for j in range(num_targets):
                    if ap_sol[i,j] == 1:
                        stage_cost[i] = dists_matrix[i][j]
                        stage_cluster[j,stage-1] = target_clusters[i, opt_i]

            stage_sum_costs[stage] = sum(stage_cost)
            stage_costs[stage] = stage_cost
            stage_clusters[stage] = stage_cluster

        opt_stage = min(stage_sum_costs, key=stage_sum_costs.get)

        for i in range(num_targets):
            curr_cost[i] = stage_costs[opt_stage][i]
            for j in range(num_views):
                target_clusters[i,j] = stage_clusters[opt_stage][i,j]

        finished_index.append(opt_stage)
        unfinished_index.remove(opt_stage)
    
    token = True
    ret_token = 0
    improve_ori_stage = []
    improve_ori_loop = []
    improve_stage = []
    improve_loop = []

    stage_iter_cluster = np.zeros((num_targets, num_views))
    for i in range(num_targets):
        for j in range(num_views):
            stage_iter_cluster[i,j] = target_clusters[i,j]
            
    curr_cost_sum = cost_calculator(stage_iter_cluster, raw_data_dict)
    original_cost = curr_cost_sum
    curr_loop_cost = curr_cost_sum

    while token:
        for stage in all_stages:
            dists_matrix_after = np.zeros((num_targets,num_targets))
            stages = list(range(num_views))
            stages.remove(stage)
            for i in range(num_targets):
                for j in range(num_targets):
                    for k in stages:
                        clu_index = np.where(target_clusters[:, k] == i)
                        curr_dist_dict = raw_data_dict[(stage+1, k+1)]

                        col_index = int(clu_index[0])
                        dists_matrix_after[j,i] += curr_dist_dict[j][col_index]
            dists_matrix_after_l = dists_matrix_after.tolist()

            ap_sol, total_cost = assignment_problem(dists_matrix_after_l, num_targets)

            for i in range(num_targets):
                for j in range(num_targets):
                    if ap_sol[i,j] == 1:
                        stage_iter_cluster[i,stage] = j
                         
            stage_iter_cost = cost_calculator(stage_iter_cluster, raw_data_dict)
            improve_ori_stage.append((original_cost - stage_iter_cost)/original_cost)
            improve_stage.append((curr_cost_sum - stage_iter_cost)/curr_cost_sum)
            curr_cost_sum = stage_iter_cost
            ret_token += 1
        
        loop_iter_cost = cost_calculator(stage_iter_cluster, raw_data_dict)
        improve_ori_loop.append((original_cost - loop_iter_cost)/original_cost)
        improve_loop.append((curr_loop_cost - loop_iter_cost)/curr_loop_cost)
        
        if loop_iter_cost >= curr_loop_cost:
            token = False
        else:
            curr_loop_cost = loop_iter_cost

            for i in range(num_targets):
                for j in range(num_views):
                    target_clusters[i,j] = stage_iter_cluster[i,j]
            
    return target_clusters

# Function for computing the total distance of a cluster under clique-based formulation
def cost_calculator(cluster, dic):
    cost = 0
    q = len(cluster)
    n = len(cluster[0])
    reshaped_cluster = cluster.T.reshape(cluster.shape[0]*cluster.shape[1])

    for i in range(q):
        sub_cluster = np.array(np.where(reshaped_cluster==i))[0]
        sub_cluster = np.array([sub_cluster[j]-j*q for j in range(n)])
        for i in range(len(sub_cluster)): # number of views
            for j in range(i+1, len(sub_cluster)):
                cost += dic[(i+1, j+1)][int(sub_cluster[i])][int(sub_cluster[j])]
    return cost

# Testing for single-source Algorithms
# User may alter dimensions for all single-source methods:
#    num_views -> number of sensors/stages
#    num_targets -> number of targets
#    filePath -> relative path for input data
#             -> 'dist_data/'        : refers to data with all uniform distributions with scale 0-100.
#             -> 'poisson_dist_1/'   : refers to data with all 4 sources of uniform distributions with scale 
#                                      0-100, and 1 source of poisson distribution with mean = 50
#             -> 'uniform_dist_1/'   : refers to data with all 4 sources of uniform distributions with scale 
#                                      0-100, and 1 source of uniform distributions with scale 0-10.
#             -> 'uniform_dist_2/'   : refers to data with all 3 sources of uniform distributions with scale 
#                                      0-100, and 2 source of uniform distributions with scale 0-10.
#             -> 'further_test_dist/': refers to data with all niform distributions with scale 0-100 with 
#                                      relatively large dimensions (over 30 sensors/stages, 30 targets).
# Please choose to un/comment the print statement correspond to the methods users may want to review
def main():
    num_views = 5
    num_targets = 6

    # Reading in the input data distance dictionary
    filePath = 'dist_data/'
    #filePath = 'uniform_dist_1/'
    #filePath = 'uniform_dist_2/'
    #filePath = 'poisson_dist_1/'
    #filePath = 'further_test_dist/'
    index = 1
    fileName = str(num_views) + 'D' + str(num_targets) + '-' + str(index) + '.dat'
    completeFileName = os.path.join(filePath, fileName)
    num_targets, num_views, raw_data_dict = data_readin.data_readin(completeFileName)

    # Test FHA under clique-based formulation
    test_forward_clu, total_cost = forward_process(raw_data_dict, num_targets, num_views)
    #print(test_forward_clu, sum(total_cost))

    # Test MSRA under clique-based formulation
    test_output_clu, total_cost = MSRA(raw_data_dict, num_targets, num_views)
    #print(test_output_clu, sum(total_cost))

    # Test RMSRA under clique-based formulation
    test_output_clu = RMSRA(raw_data_dict, num_targets, num_views)
    total_cost = cost_calculator(test_output_clu, raw_data_dict)
    #print(test_output_clu, total_cost)

    # Test MSRA under path-based formulation
    test_output_clu = MSRA_p(raw_data_dict, num_targets, num_views)
    #print(test_output_clu)
    
    # Test for LP as benchmarks
    #lps = solver.LP_Solution(num_views, num_targets, raw_data_dict)
    #m = lps.solver()
    #lp_cluster = lps.getPath(m)
    #lp_cost = cost_calculator(lp_cluster, raw_data_dict)

if __name__ == "__main__":
    main()