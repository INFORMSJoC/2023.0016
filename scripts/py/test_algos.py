import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import data_readin
import single_source_algo as ssa
import lmvc_algo as lmvc
import cbsnmf_algo as cbsnmf
import matplotlib.pyplot as plt
import time
import LP_solver as solver
import xlsxwriter
import pandas as pd

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

# Function to test all the single-source methods at the same time.
# User may alter inputs for all single-source methods:
#    num_views -> number of sensors/stages
#    num_targets -> number of targets
#    num_generated -> number of data sources (further used in multi-source section)
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
# Output: single-source clustering information.
# Note: 
#    Under medium dimensions (10 sensors/stages, 20targets), LP will face computationally difficulties.
#    Under large dimensions (over 30 sensors/stages, 30 targets), LP will not be able to 
#    solve in reasonalbe time, and RMSRA will take longer time to solve. 
#    Please comment out the LP section in the function for large dimensions.
def test_single_source(num_views, num_targets, num_generated):
    test_output_clu_dict = {}

    forward_time = 0
    rmsra_time = 0
    lp_time = 0

    forward_cost = 0
    rmsra_cost = 0
    lp_cost = 0

    for i in range(num_generated):
        index = i + 1
        filePath = 'dist_data/'
        #filePath = 'uniform_dist_1/'
        #filePath = 'uniform_dist_2/'
        #filePath = 'poisson_dist_1/'
        #filePath = 'further_test_dist/'
        fileName = str(num_views) + 'D' + str(num_targets) + '-' + str(index) + '.dat'
        completeFileName = os.path.join(filePath, fileName)
        num_targets, num_views, raw_data_dict = data_readin.data_readin(completeFileName)

        time_one_start = time.time()

        # FHA
        test_forward_clu, total_cost = ssa.forward_process(raw_data_dict, num_targets, num_views)
        forward_time += time.time()-time_one_start
        temp = cost_calculator(test_forward_clu, raw_data_dict)
        forward_cost += temp

        # RMSRA
        test_output_clu = ssa.RMSRA(raw_data_dict, num_targets, num_views)
        test_output_clu = test_output_clu.astype(int)
        test_output_clu_dict[i] = test_output_clu

        temp = cost_calculator(test_output_clu, raw_data_dict)
        rmsra_cost += temp
        rmsra_time += time.time() - time_one_start
        
        # LP
        time_comparison_start = time.time()
        lps = solver.LP_Solution(num_views, num_targets, raw_data_dict)
        m = lps.solver()
        lp_time += time.time() - time_comparison_start
        lp_cluster = lps.getPath(m)
        lp_cost += cost_calculator(lp_cluster, raw_data_dict)


    print("The RHA clique-based cost is:", forward_cost / num_generated)
    print("The RMSRA clique-based cost is:", rmsra_cost / num_generated)
    print("The LP clique-based cost is:", lp_cost / num_generated)

    print('\nFHA time cost %.5f s' % (forward_time / num_generated))
    print('RMSRA time cost %.5f s' % (rmsra_time / num_generated))
    print('LP time cost %.5f s' % (lp_time / num_generated))
    
    return test_output_clu_dict

# Function to test all the LMVC algorithm.
# User may alter inputs for LMVC algorithm:
#    num_views -> number of sensors/stages
#    num_targets -> number of targets
#    num_generated -> number of data sources
#    test_output_clu_dict -> single-source clustering information
#    filePath -> relative path for input data (Please refer to the test_single_source() description)
# Note:
#    All the input information should match with the outputs of the single-source methods. 
def test_LMVC(test_output_clu_dict, num_views, num_targets, num_generated):
    lmvc_time = 0
    lmvc_input_dict = {}

    for i in range(num_generated): 
        lmvc_input = np.ones((num_targets, num_views))
        for p in range(num_targets):
            for q in range(num_targets):
                for r in range(num_views):
                    if test_output_clu_dict[i][q][r] == p:
                        lmvc_input[p][r] = q
        lmvc_input = lmvc_input.astype(int)
        lmvc_input_dict[i] = lmvc_input

    lmvc_start = time.time()

    lmvc_results = []
    lmvc_cost = 0
    cross_domain_source = lmvc.Data(num_views, num_targets, lmvc_input_dict)
    cluster = cross_domain_source.process()
    cluster = np.array(cluster)
    cluster_rev = np.ones([num_targets, num_views])
    for p in range(len(cluster_rev)):
        for q in range(len(cluster_rev[0])):
            row_num = cluster[p][q]
            col_num = q
            cluster_rev[row_num,col_num] = p
    cluster_rev = cluster_rev.astype(int)

    lmvc_time += time.time() - lmvc_start

    for i in range(num_generated):
        index = i + 1
        filePath = 'dist_data/'
        #filePath = 'uniform_dist_1/'
        #filePath = 'uniform_dist_2/'
        #filePath = 'poisson_dist_1/'
        #filePath = 'further_test_dist/'
        fileName = str(num_views) + 'D' + str(num_targets) + '-' + str(index) + '.dat'
        completeFileName = os.path.join(filePath, fileName)
        num_targets, num_views, raw_data_dict = data_readin.data_readin(completeFileName)

        lmvc_cost += cost_calculator(cluster_rev, raw_data_dict)
    lmvc_results.append(lmvc_cost/num_generated)

    print("\nThe LMVC clique-based cost  is:", np.mean(lmvc_results))
    print('LMVC time cost %.5f s' % (lmvc_time))
    
    return cluster_rev

# Function to test all the CBSNMF algorithm.
# User may alter inputs for CBSNMF algorithm:
#    num_views -> number of sensors/stages
#    num_targets -> number of targets
#    num_generated -> number of data sources
#    test_output_clu_dict -> single-source clustering information
#    filePath -> relative path for input data (Please refer to the test_single_source() description)
#    connected_reg -> penality factor for targets clustered in the same group
#    unconeected_reg -> penality factor for targets not clustered in the same group
#    alpha -> coefficient for the regularization matrix
#    max_iter -> maximum iteration for cbsnmf
# Note:
#    Input information should match with the outputs of the single-source methods. 
def test_cbsnmf(test_output_clu_dict, num_views, num_targets, num_generated):
    connected_reg = 0.1
    unconeected_reg = 100
    test_alphas = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1, 1, 10, 100, 1000]
    max_iter =1000
        
    test_cbsnmf = []
    test_cbsnmf_times = []
    test_cbsnmf_results = []
    test_alpha = []
    for a in test_alphas:
        test_cbsnmf_costs = []
        cbsnmf_results = []
        for k in range(5):
            cbsnmf_cost = 0
            cbsnmf_time = 0
            cbsnmf_time_start = time.time()    

            # choose the types of similarity matrix
            #indicator_matrix = cbsnmf.get_norm_indicator_matrix(test_output_clu_dict, num_targets, num_views, num_generated)
            indicator_matrix = cbsnmf.get_scale_indicator_matrix(test_output_clu_dict, num_targets, num_views)
            #indicator_matrix = cbsnmf.density_indicator_matrix(test_output_clu_dict, num_targets, num_views, num_generated)
            
            # choose the tyoe of regularization matrix
            #g = cbsnmf.get_reg_matrix(indicator_matrix, connected_reg, unconeected_reg)
            g = cbsnmf.get_scale_reg_matrix(test_output_clu_dict, num_targets, num_views)

            u = cbsnmf.matrix_fact(indicator_matrix, g, a, max_iter, num_targets, num_views)
            cbsnmf_result = cbsnmf.get_concensus_matrix(u, num_targets, num_views)
            cbsnmf_results.append(cbsnmf_result)

            cbsnmf_time += time.time() - cbsnmf_time_start

            for i in range(num_generated):
                index = i + 1
                filePath = 'dist_data/'
                #filePath = 'uniform_dist_1/'
                #filePath = 'uniform_dist_2/'
                #filePath = 'poisson_dist_1/'
                #filePath = 'further_test_dist/'
                fileName = str(num_views) + 'D' + str(num_targets) + '-' + str(index) + '.dat'
                completeFileName = os.path.join(filePath, fileName)
                num_targets, num_views, raw_data_dict = data_readin.data_readin(completeFileName)

                cbsnmf_cost += cost_calculator(cbsnmf_result, raw_data_dict)
        
            test_cbsnmf_costs.append(cbsnmf_cost/num_generated)
        test_cbsnmf_times.append(cbsnmf_time/5)
        test_cbsnmf_results.append(cbsnmf_results[test_cbsnmf_costs.index(min(test_cbsnmf_costs))])
        test_cbsnmf.append(min(test_cbsnmf_costs))
        test_alpha.append(a)

    output_cbsnmf_cost = min(test_cbsnmf)
    output_cbsnmf_result = test_cbsnmf_results[test_cbsnmf.index(min(test_cbsnmf))]
    output_cbsnmf_alpha = test_alpha[test_cbsnmf.index(min(test_cbsnmf))]
    output_cbsnmf_time = test_cbsnmf_times[test_cbsnmf.index(min(test_cbsnmf))]
    print("\nThe CBSNMF clique-based cost  is:", output_cbsnmf_cost)
    print('The alpha of CBSNMF is: ', output_cbsnmf_alpha)
    print('The CBSNMF time cost %.5f s' % output_cbsnmf_time)
    
    return output_cbsnmf_result.astype(int)

# Testing for all the methods and algorithms.
# User may alter inputs for methods and algorithms:
#    num_views -> number of sensors/stages
#    num_targets -> number of targets
#    num_generated -> number of data sources
def main():
    num_views = 5
    num_targets = 6
    num_generated = 5
    test_output_clu_dict = test_single_source(num_views, num_targets, num_generated)
    
    output_clu = test_cbsnmf(test_output_clu_dict, num_views, num_targets, num_generated)
    print(output_clu)
    
    #output_clu = test_LMVC(test_output_clu_dict, num_views, num_targets, num_generated)
    #print(output_clu)
    


if __name__ == "__main__":
    main()