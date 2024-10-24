import gurobipy as gp
from gurobipy import GRB
import random as rand
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
import itertools
from numpy import random
import LP_solver as solver
from scipy.optimize import linear_sum_assignment

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

# Function for computing the total distance of a cluster under path-based formulation
def path_cost_calculator(inptu_matrix, raw_data_dict, num_targets, num_views):
    clus_dist = np.zeros((num_targets))
    for i in range(num_views-1):
        for m in range(num_targets):
            curr_tar = inptu_matrix[m, i]
            for n in range(num_targets):
                if inptu_matrix[n, i+1] == curr_tar:
                    curr_dist = raw_data_dict[(i+1,i+2)][m,n]
                    clus_dist[m] += curr_dist
    return sum(clus_dist)

# Function for computing the total distance of a cluster under clique-based formulation
def cost_calculator(cluster, dic):
    cost = 0
    q = len(cluster)
    n = len(cluster[0])
    reshaped_cluster = cluster.T.reshape(cluster.shape[0]*cluster.shape[1])
    
    for i in range(q):
        sub_cluster = np.array(np.where(reshaped_cluster==(i+1)))[0]
        sub_cluster = np.array([sub_cluster[j]-j*q for j in range(n)])
        for i in range(len(sub_cluster)): # number of views
            for j in range(i+1, len(sub_cluster)):
                cost += dic[(i+1, j+1)][int(sub_cluster[i])][int(sub_cluster[j])]
    return cost

# Function to generate an arbitary True Clusering Matrix for all concensus.
# User may alter dimensions for the methods:
#    num_views -> number of sensors/stages
#    num_targets -> number of targets
def Generate_True_Clus_Matrix(num_targets, num_views):
    true_clus_matrix = np.zeros((num_targets, num_views))

    for i in range(num_views):
        target_array = np.arange(1,num_targets+1)
        for j in range(num_targets):
            position = rand.randint(0,len(target_array)-1)
            true_clus_matrix[j,i] = target_array[position]
            target_array = np.delete(target_array, position)
    
    return true_clus_matrix

# Function to Generate an arbitary True Position Matrix for all concensus based on the True Clustering Matrix, 
# under the scales that fit Uniorm(0,100).
# Note:
#    num_views and num_targets will have to match with True Clustering Matrix. 
def Generate_True_Position_Matrix(true_clus_matrix, num_targets, num_views):
    true_position_matrix = np.zeros((num_targets, num_views))
    max_scale = 100
    base_scale = max_scale/num_targets
    for i in range(num_targets):
        for j in range(num_views):
            curr_scale =  true_clus_matrix[i,j]
            lower_bound = (curr_scale-1)*base_scale
            upper_bound = curr_scale*base_scale

            position = np.random.randint(lower_bound,upper_bound)
            true_position_matrix[i,j] = position
    return true_position_matrix

# Function to Generate the Distance Matrix and Dictionary based on the Position Matrix.
def Generate_Dist_Dict(position_matrix, num_targets, num_views):
    dist_matrices = {}

    for m in range(num_views):
        for n in range(m+1, num_views):
            dist_matrix = np.zeros((num_targets, num_targets))
            for i in range(num_targets):
                for j in range(num_targets):
                    dist_matrix[i,j] = abs(position_matrix[i,m] - position_matrix[j,n])
            dist_matrices[(m+1,n+1)] = dist_matrix
            dist_matrices[(n+1,m+1)] = dist_matrix.T
    
    return dist_matrices

# Function to add Noise to one sensor among all the sensors .
# Note:
#    Noise follows Normal Distribution with mean=0 and std=noise_scale*position.
#    noise_scale should be [0,1]
def Add_Sensor_Noise(position_matrix, noise_scale, num_targets, num_views):
    noise_sensor = rand.randint(0,num_views-1)
    noise_position_matrix = np.zeros((num_targets, num_views))

    for i in range(num_targets):
        for j in range(num_views):
            if j != noise_sensor:
                noise_position_matrix[i,j] = position_matrix[i,j]
            else:
                target_position = position_matrix[i, noise_sensor]
                position_noise = int(np.random.normal(loc=0, scale=target_position*noise_scale))
                noise_position_matrix[i,j] = target_position + position_noise
                
    return noise_position_matrix

# Function to test generating True Position Matrix.
def test_generate_true_clus_matrix(num_targets, num_views):
    true_clus_matrix = Generate_True_Clus_Matrix(num_targets, num_views)
    return true_clus_matrix

# Function to test generating True Distance Matrix.
def test_generate_true_pos_matrix(true_clus_matrix, num_targets, num_views):
    true_position_matrix = Generate_True_Position_Matrix(true_clus_matrix, num_targets, num_views)
    return true_position_matrix

# Function to test generating Distance Dicctionary.
def test_generate_dist_dict(position_matrix, num_targets, num_views):
    dist_dict = Generate_Dist_Dict(position_matrix, num_targets, num_views)
    return dist_dict

# Function to test adding Noise.
def test_add_sensor_noise(true_position_matrix, noise_scale, num_targets, num_views):
    noise_position_matrix = Add_Sensor_Noise(true_position_matrix, noise_scale, num_targets, num_views)
    return noise_position_matrix

# match the first sensor clustering index with the true cluster matrix
def match_clu_indicator(input_cluster, true_clus_matrix, num_targets, num_views):
    match_dict = {}
    for i in range(num_targets):
        match_dict[input_cluster[i,0]] = true_clus_matrix[i,0]

    output_matrix = np.zeros((num_targets, num_views))
    for i in range(num_targets):
        for j in range(num_views):
            output_matrix[i,j] = match_dict[input_cluster[i,j]]
    return output_matrix

# Function to solve the Clique-based formulation single-source problem under noisy setting
def clique_based_single_source_noise_test(noise_dist_dict, true_clus_matrix, num_targets, num_views):
    clique_output_clu = ssa.RMSRA(noise_dist_dict, num_targets, num_views)
    clique_output_matrix = match_clu_indicator(clique_output_clu, true_clus_matrix, num_targets, num_views)
    return clique_output_matrix

# Function to solve the Path-based formulation single-source problem under noisy setting
def path_based_single_source_noise_test(noise_dist_dict, true_clus_matrix, num_targets, num_views):
    path_output_clu = ssa.MSRA_p(noise_dist_dict, num_targets, num_views)
    path_output_matrix = match_clu_indicator(path_output_clu, true_clus_matrix, num_targets, num_views)
    return path_output_matrix

# Function to test Single Source Single Noise sensor
# User can test a sequence of combination of dimensions.
# User should alter the input dimension for the single source case to test
#    num_targets_lower -> lower bound for the number of targets
#    num_views_lower -> lower bound for the number of sensors/stages
#    num_targets_upper -> upper bound for the number of targets
#    num_views_upper -> upper bound for the number of sensors/stages
#    noise_scales -> noise scaling vector, each scale should be in [0,1]
# Clique-based/path-based formualtion can be tested based on function used
#     un/comment to obtain the results user wanted.
def single_source_noise_test(num_targets_lower, num_targets_upper, num_views_lower, num_views_upper, noise_scales):
    noise_dict = {}

    for s in noise_scales:   
        noise_indicators = []
        for i in range(num_targets_lower, num_targets_upper):
            noise_indicator = []
            for j in range(num_views_lower, num_views_upper):  
                true_clus_matrix = Generate_True_Clus_Matrix(i, j)
                true_position_matrix = Generate_True_Position_Matrix(true_clus_matrix, i, j)
                true_dist_dict = Generate_Dist_Dict(true_position_matrix, i, j)

                noise_position_matrix = Add_Sensor_Noise(true_position_matrix, s, i, j)
                noise_dist_dict = Generate_Dist_Dict(noise_position_matrix, i, j)

                # choose path-based or cliqeu-baed formulation
                # path-based formulation
                noise_output_clu = ssa.MSRA_p(noise_dist_dict, i, j)
                
                # clique-based formualtion
                #noise_output_clu = ssa.RMSRA(noise_dist_dict, i, j)
                
                noise_output_matrix = match_clu_indicator(noise_output_clu, true_clus_matrix, i, j)

                # choose cost computing series
                # path-based cost computing series
                true_cost = path_cost_calculator(true_clus_matrix, true_dist_dict, i, j)
                noise_cost = path_cost_calculator(noise_output_matrix, true_dist_dict, i, j)
                opt_gap = round((noise_cost - true_cost) / true_cost * 100, 3)
                
                # clique-based cost computing series
                #true_cost = cost_calculator(true_clus_matrix, true_dist_dict)
                #noise_cost = cost_calculator(noise_output_matrix, true_dist_dict)
                #opt_gap = round((noise_cost - true_cost) / true_cost * 100, 3)

                noise_indicator.append(opt_gap)
            noise_indicators.append(noise_indicator)

        if s not in noise_dict.keys():
            noise_dict[s] = [noise_indicators]
        else:
            noise_dict[s].append(noise_indicators)
    return noise_dict

# Function to test Multi-source Noise sensor under CBSNMF
# User can test a sequence of combination of dimensions.
# User should alter the input dimension for the single source case to test
#    num_targets_lower -> lower bound for the number of targets
#    num_views_lower -> lower bound for the number of sensors/stages
#    num_targets_upper -> upper bound for the number of targets
#    num_views_upper -> upper bound for the number of sensors/stages
#    noise_scales -> noise scaling vector, each scale should be in [0,1]
# User may un/comment to choose similarity matrix and regularization matrix
# User may output only with clusters, costs, alphas, and maximum iteration information 
# through corresponding dictionary
def CBSNMF_noise_test(num_targets_lower, num_targets_upper, num_views_lower, num_views_upper, num_generated, noise_scales):
    test_alphas = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2, 1e-1, 1, 10, 100, 1000]
    test_iters = [10, 15, 20, 25, 50, 75, 100, 150, 200, 500, 750, 1000, 5000]
    connected_reg = 0.1
    unconeected_reg = 100

    opt_noise_clus_dic = {}
    opt_noise_costs_dic = {}
    opt_noise_alphas_dic = {}
    opt_noise_iters_dic = {}
    opt_gap_dic = {}
    for s in noise_scales:
        opt_noise_clus = []
        opt_noise_costs = []
        opt_noise_alphas = []
        opt_noise_iters = []
        for i in range(num_targets_lower, num_targets_upper):
            opt_noise_clu = []
            opt_noise_cost = []
            opt_noise_alpha = []
            opt_noise_iter = []
            for j in range(num_views_lower,num_views_upper):
                true_clus_matrix = Generate_True_Clus_Matrix(i, j)
                true_position_matrix = Generate_True_Position_Matrix(true_clus_matrix, i, j)
                true_dist_dict = Generate_Dist_Dict(true_position_matrix, i, j)

                single_source_outputs = {}
                single_source_matrixs = {}
                for k in range(num_generated):
                    noise_position_matrix = Add_Sensor_Noise(true_position_matrix, s, i, j)
                    noise_dist_dict = Generate_Dist_Dict(noise_position_matrix, i, j)
                    noise_output_clu = ssa.RMSRA(noise_dist_dict, i, j)
                    
                    noise_output_matrix = match_clu_indicator(noise_output_clu, true_clus_matrix, i, j)

                    single_source_outputs[k] = noise_output_clu
                    single_source_matrixs[k] = noise_output_matrix

                optimal_con_cost = 999999
                for a in test_alphas:
                    for max_iter in test_iters:
                        for k in range(5):
                            #indicator_matrix = cbsnmf.get_norm_indicator_matrix(single_source_outputs, i, j, num_generated)
                            indicator_matrix = cbsnmf.get_scale_indicator_matrix(single_source_outputs, i, j)
                            #indicator_matrix = cbsnmf.density_indicator_matrix(single_source_outputs, i, j, num_generated)

                            #g = cbsnmf.get_reg_matrix(indicator_matrix, connected_reg, unconeected_reg)
                            g = cbsnmf.get_scale_reg_matrix(single_source_outputs, i, j)

                            u = cbsnmf.matrix_fact(indicator_matrix, g, a, max_iter, i, j)
                            cbsnmf_result = cbsnmf.get_concensus_matrix(u, i, j)
                            cbsnmf_matrix = match_clu_indicator(cbsnmf_result, true_clus_matrix, i, j)

                            concensus_cost = cost_calculator(cbsnmf_matrix, true_dist_dict)
                            if concensus_cost < optimal_con_cost:
                                optimal_clu = cbsnmf_matrix
                                optimal_con_cost = concensus_cost
                                optimal_alpha = a
                                optimal_iter = max_iter
                                
                opt_noise_clu.append(optimal_clu)
                opt_noise_cost.append(optimal_con_cost)
                opt_noise_alpha.append(optimal_alpha)
                opt_noise_iter.append(max_iter)
            
                true_cost = cost_calculator(true_clus_matrix, true_dist_dict)
                opt_con_cost = cost_calculator(optimal_clu, true_dist_dict)
                opt_gap = round((opt_con_cost - true_cost) / true_cost * 100, 3)
                opt_gap_dic[(s,i,j)] = (optimal_alpha, max_iter, opt_gap, optimal_clu)
            
            opt_noise_clus.append(opt_noise_clu)
            opt_noise_costs.append(opt_noise_cost)
            opt_noise_alphas.append(opt_noise_alpha)
            opt_noise_iters.append(opt_noise_iter)

        
        if s not in opt_noise_clus_dic.keys():
            opt_noise_clus_dic[s] = [opt_noise_clus]
        else:
            opt_noise_clus_dic[s].append(opt_noise_clus)
        
        if s not in opt_noise_costs_dic.keys():
            opt_noise_costs_dic[s] = [opt_noise_costs]
        else:
            opt_noise_costs_dic[s].append(opt_noise_costs)
        
        
        if s not in opt_noise_alphas_dic.keys():
            opt_noise_alphas_dic[s] = [opt_noise_alphas]
        else:
            opt_noise_alphas_dic[s].append(opt_noise_alphas)
        
        if s not in opt_noise_iters_dic.keys():
            opt_noise_iters_dic[s] = [opt_noise_iters]
        else:
            opt_noise_iters_dic[s].append(opt_noise_iters)
            
    return opt_gap_dic

# Function to test Multi-source Noise sensor under LMVC
# User can test a sequence of combination of dimensions.
# User should alter the input dimension for the single source case to test
#    num_targets_lower -> lower bound for the number of targets
#    num_views_lower -> lower bound for the number of sensors/stages
#    num_targets_upper -> upper bound for the number of targets
#    num_views_upper -> upper bound for the number of sensors/stages
#    noise_scales -> noise scaling vector, each scale should be in [0,1]
def LMVC_noise_test(num_targets_lower, num_targets_upper, num_views_lower, num_views_upper, num_generated, noise_scales):
    noise_dict = {}
    opt_gap_dic = {}
    for s in noise_scales:
        noise_indicators = []
        for i in range(num_targets_lower, num_targets_upper):
            noise_indicator = []
            for j in range(num_views_lower,num_views_upper):
                true_clus_matrix = Generate_True_Clus_Matrix(i, j)
                true_position_matrix = Generate_True_Position_Matrix(true_clus_matrix, i, j)
                true_dist_dict = Generate_Dist_Dict(true_position_matrix, i, j)

                single_source_outputs = {}
                single_source_matrixs = {}
                LMVC_inputs = {}
                
                for k in range(num_generated):
                    noise_position_matrix = Add_Sensor_Noise(true_position_matrix, s, i, j)
                    noise_dist_dict = Generate_Dist_Dict(noise_position_matrix, i, j)
                    noise_output_clu = ssa.RMSRA(noise_dist_dict, i, j)
                    
                    noise_output_matrix = match_clu_indicator(noise_output_clu, true_clus_matrix, i, j)

                    single_source_outputs[k] = noise_output_clu
                    single_source_matrixs[k] = noise_output_matrix

                    LMVC_input = np.ones((i, j))
                    for p in range(i):
                        for q in range(i):
                            for r in range(j):
                                if noise_output_clu[q][r] == p:
                                    LMVC_input[p][r] = q
                    LMVC_input = LMVC_input.astype(int)
                    LMVC_inputs[k] = LMVC_input
                    
                cross_domain_source = lmvc.Data(j, i, LMVC_inputs)
                cluster = cross_domain_source.process()
                cluster = np.array(cluster)
                cluster_rev = np.ones([i, j])
                for p in range(len(cluster_rev)):
                    for q in range(len(cluster_rev[0])):
                        row_num = cluster[p][q]
                        col_num = q
                        cluster_rev[row_num,col_num] = p
                cluster_rev = cluster_rev.astype(int)
                cluster_matrix = match_clu_indicator(cluster_rev, true_clus_matrix, i, j)

                true_cost = cost_calculator(true_clus_matrix, true_dist_dict)
                opt_con_cost = cost_calculator(cluster_matrix, true_dist_dict)
                opt_gap = round((opt_con_cost - true_cost) / true_cost * 100, 3)
                opt_gap_dic[(s,i,j)] = (opt_gap, cluster_matrix)

                noise_indicator.append(opt_gap)
            noise_indicators.append(noise_indicator)
            
        if s not in noise_dict.keys():
            noise_dict[s] = [noise_indicators]
        else:
            noise_dict[s].append(noise_indicators)
    return opt_gap_dic

def main():
    num_targets = 5
    num_views = 6
    noise_scale = 0.1

    # generate gound true cluster, position matrix and distance dictionary 
    true_clus_matrix = test_generate_true_clus_matrix(num_targets, num_views)
    true_position_matrix = test_generate_true_pos_matrix(true_clus_matrix, num_targets, num_views) 
    true_dist_dict = test_generate_dist_dict(true_position_matrix, num_targets, num_views)

    # generate noise position matrix and distance dictionary 
    noise_position_matrix = test_add_sensor_noise(true_position_matrix, noise_scale, num_targets, num_views)
    noise_dist_dict = test_generate_dist_dict(noise_position_matrix, num_targets, num_views)

    #print('True Posittion Matrix:')
    #print(true_position_matrix)
    #print('Noise Posittion Matrix:')
    #print(noise_position_matrix)
    #print('Matrix Difference:')
    #print(noise_position_matrix - true_position_matrix)
    
    # solve the single-source noise problem under clique-based formulation and compare the optimality gaps
    clique_output_matrix = clique_based_single_source_noise_test(noise_dist_dict, true_clus_matrix, num_targets, num_views)
    true_cost = cost_calculator(true_clus_matrix, true_dist_dict)
    noise_cost = cost_calculator(clique_output_matrix, true_dist_dict)
    opt_gap = round((noise_cost - true_cost) / true_cost * 100, 3)

    #print('Clique-based Formulation')
    #print("True Clustering Distance:")
    #print(true_cost)
    #print("Noise Clustering Distance:")
    #print(noise_cost)
    #print('Optimality Gap: ')
    #print(str(opt_gap) + '%')
    
    # solve the single-source noise problem under path-based formaltion and compare the optimality gaps
    path_output_matrix = path_based_single_source_noise_test(noise_dist_dict, true_clus_matrix, num_targets, num_views)
    true_cost = path_cost_calculator(true_clus_matrix, true_dist_dict, num_targets, num_views)
    noise_cost = path_cost_calculator(path_output_matrix, true_dist_dict, num_targets, num_views)
    opt_gap = round((noise_cost - true_cost) / true_cost * 100, 3)
    
    #print('Path-based Formulation')
    #print("True Clustering Distance:")
    #print(true_cost)
    #print("Noise Clustering Distance:")
    #print(noise_cost)
    #print('Optimality Gap: ')
    #print(str(opt_gap) + '%')
    
    # input variables for single-source and multi-source noise testing
    num_targets_lower = 3
    num_targets_upper = 11
    num_views_lower = 3
    num_views_upper = 21
    num_generated = 5
    noise_scales = [s/20 for s in range(20)]
    
    # test on the single source noise based on different noise scaling
    single_noise = single_source_noise_test(num_targets_lower, num_targets_upper, num_views_lower, num_views_upper, noise_scales)
    #print(single_noise)
    
    # test on the multi-source noise under LMVC based on different noise scaling
    lmvc_noise = LMVC_noise_test(num_targets_lower, num_targets_upper, num_views_lower, num_views_upper, num_generated, noise_scales)
    #print(lmvc_noise)
    
    # test on the multi-source noise under CBSNMF based on different noise scaling
    cbsnmf_noise = CBSNMF_noise_test(num_targets_lower, num_targets_upper, num_views_lower, num_views_upper, num_generated, noise_scales)
    #print(cbsnmf_noise)
    
if __name__ == "__main__":
    main()