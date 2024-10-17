import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
import data_readin
import matplotlib.pyplot as plt
import time
import LP_solver as solver
import xlsxwriter
import itertools
from numpy import random
from scipy.optimize import linear_sum_assignment

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

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

# Function to generate the similarity matrix from each single source clustering results
def get_indicator_matrix(input_source, num_targets, num_views):
    indicator_matrix = np.zeros((num_targets*num_views, num_targets*num_views))

    for i in range(num_targets*num_views):
        indicator_set = input_source[:,int(i/num_targets)]
        indicator_sectors = []
        for j in range(num_views):
            indicator_sector = np.zeros((num_targets)).tolist()
            for k in range(num_targets):
                if input_source[k,j] == indicator_set[i%num_targets]:
                    indicator_sector[k] = 1
                    indicator_sectors.append(indicator_sector)
        indicator_sectors = np.array(list(itertools.chain(*indicator_sectors))).astype(int)
        indicator_matrix[i,:] = indicator_sectors

    indicator_matrix = indicator_matrix.astype(int)
    return indicator_matrix

# Function to generate the normalized similarity matrix among all the single-source clustering results
def get_norm_indicator_matrix(input_data_dict, num_targets, num_views, num_source):
    input_keys = input_data_dict.keys()
    indicator_matrices = []
    for k in input_keys:
        input_source = input_data_dict[k]
        indicator_matrix = get_indicator_matrix(input_source, num_targets, num_views)
        indicator_matrices.append(indicator_matrix)
        
    indicator_matrix = sum(indicator_matrices)
    imax, jmax = indicator_matrix.shape
    indicator_matrix = indicator_matrix.astype('float64')
    for i in range(imax):
        for j in range(jmax):
            if indicator_matrix[i,j] != 0:
                indicator_matrix[i,j] = round(float((indicator_matrix[i,j]/num_source)),3)
            else:
                indicator_matrix[i,j] = 0
    return indicator_matrix

# Function to generate the binary similarity matrix among all the single-source clustering results
def get_scale_indicator_matrix(input_data_dict, num_targets, num_views):
    input_keys = input_data_dict.keys()
    indicator_matrices = []
    for k in input_keys:
        input_source = input_data_dict[k]
        indicator_matrix = get_indicator_matrix(input_source, num_targets, num_views)
        indicator_matrices.append(indicator_matrix)

    indicator_matrices = sum(indicator_matrices)
    imax, jmax = indicator_matrices.shape

    for i in range(imax):
        for j in range(jmax):
            if indicator_matrices[i,j] > 0:
                indicator_matrices[i,j] = 1
    return indicator_matrices

# Function to generate the density-based similarity matrix among all the single-source clustering results
def density_indicator_matrix(input_data_dict, num_targets, num_views, num_source):
    indicator_matrices = []
    for i in range(5):
        input_source = input_data_dict[i]
        indicator_matrix = get_indicator_matrix(input_source, num_targets, num_views)
        indicator_matrices.append(indicator_matrix)
        
    indicator_matrix = sum(indicator_matrices)
    imax, jmax = indicator_matrix.shape
    indicator_matrix = indicator_matrix.astype('float64')
    
    for i in range(imax):
        for j in range(jmax):
            if indicator_matrix[i,j] == 0:
                indicator_matrix[i,j] = 0
            else:
                indicator_matrix[i,j] = round(float(1/(indicator_matrix[i,j]/num_source)),3)

    return indicator_matrix

# Function to get the regularized matrix from the indicator matrix
# connected_reg:   penality factor for targets clustered in the same group
# unconeected_reg: penality factor for targets not clustered in the same group
# connected_reg << unconeected_reg
def get_reg_matrix(indicator_matrix, connected_reg, unconeected_reg):
    matrix_dim = len(indicator_matrix)
    reg_matrix = np.zeros((matrix_dim, matrix_dim))
    for i in range(matrix_dim):
        for j in range(matrix_dim):
            if indicator_matrix[i,j] > 0:
                reg_matrix[i,j] = connected_reg
            else:
                reg_matrix[i,j] = unconeected_reg
    return reg_matrix

# Function to get the scaled regularized matrix from the indicator matrix
# the penality factor for connected two targets will be 0
# the penality factor for unconnnected two targets will be 
# the maximum {2/indicator_matrices[i,j]} if ndicator_matrices[i,j] != 0
def get_scale_reg_matrix(input_data_dict, num_targets, num_views):
    input_keys = input_data_dict.keys()
    indicator_matrices = []
    for k in input_keys:
        input_source = input_data_dict[k]
        indicator_matrix = get_indicator_matrix(input_source, num_targets, num_views)
        indicator_matrices.append(indicator_matrix)
    indicator_matrices = sum(indicator_matrices)
    imax, jmax = indicator_matrices.shape

    scale_indicators = []
    for i in range(imax):
        scale_indi = []
        for j in range(jmax):
            if indicator_matrices[i,j] > 0:
                scale_indi.append(round(float(1/indicator_matrices[i,j]),3))
            else:
                scale_indi.append(0)
        scale_indicators.append(scale_indi)
    scale_indicators = np.array(scale_indicators)

    imax, jmax = scale_indicators.shape
    penality_fac = np.amax(scale_indicators) * 2
    for i in range(imax):
        for j in range(jmax):
            if scale_indicators[i,j] == 0:
                scale_indicators[i,j] = penality_fac
    return scale_indicators

# Function to perfrom NMF
# indicator_matrix: similarity matrix for factorization
# reg_matrix: regularization matrix for cbsnmf
# alpha:  coefficient for the regularization matrix
# max_iter: maximum iteration for cbsnmf
# num_targets: number of targets
# num_views: number of sensors/stages
def matrix_fact(indicator_matrix, reg_matrix, alpha, max_iter, num_targets, num_views):
    u = random.randint(100, size=(num_targets*num_views,num_targets))
    for i in range(max_iter):
        numeroator = indicator_matrix@u
        denominator = u@u.T@u + alpha*reg_matrix@u@u.T@reg_matrix@u
        u = (numeroator/(denominator+1e-9))*u
    return u

# Function to get the concensus matrix from the output of the matrix factorization
# u: output for the matrix factorization/Indicator matrix in paper
# num_targets: number of targets
# num_views: number of sensors/stages
def get_concensus_matrix(u, num_targets, num_views):
    output_res = np.zeros((num_targets, num_views))
    for i in range(num_views):
        row_ind, col_ind = linear_sum_assignment(-u[i*num_targets:(i+1)*num_targets,:])
        for j in range(num_targets):
            output_res[j,i] = col_ind[j]
    return output_res

# Testing for CBSNMF Algorithms
# User may alter dimensions for CBSNMF methods:
#    num_views -> number of sensors/stages
#    num_targets -> number of targets
#    num_source -> number of data sources
#    input_data_dict -> outputs from single-source methods
#    connected_reg -> penality factor for targets clustered in the same group
#    unconeected_reg -> penality factor for targets not clustered in the same group
#    alpha -> coefficient for the regularization matrix
#    max_iter -> maximum iteration for cbsnmf
# Please choose to un/comment the print statement correspond to indicator and regularization matrix
# users may want to review.
def main():
    num_targets = 5
    num_views = 3
    num_source = 5
    input_data_dict = {0: np.array([[0, 3, 4],
                           [1, 1, 1],
                           [2, 2, 2],
                           [3, 0, 3],
                           [4, 4, 0]]),
                       1: np.array([[0, 2, 4],
                           [1, 0, 1],
                           [2, 1, 3],
                           [3, 4, 2],
                           [4, 3, 0]]),
                       2: np.array([[0, 2, 2],
                           [1, 3, 3],
                           [2, 0, 1],
                           [3, 4, 4],
                           [4, 1, 0]]),
                       3: np.array([[0, 1, 0],
                           [1, 2, 1],
                           [2, 4, 3],
                           [3, 3, 2],
                           [4, 0, 4]]),
                       4: np.array([[0, 1, 0],
                           [1, 0, 2],
                           [2, 4, 3],
                           [3, 2, 4],
                           [4, 3, 1]])}
    
    connected_reg = 0.1
    unconeected_reg = 10
    alpha = 10
    max_iter = 1000
    
    # choose type of the similarity matrix
    #indicator_matrix = get_norm_indicator_matrix(input_data_dict, num_targets, num_views, num_source)
    indicator_matrix = get_scale_indicator_matrix(input_data_dict, num_targets, num_views)
    #indicator_matrix = density_indicator_matrix(input_data_dict, num_targets, num_views, num_source)
    
    # choose type of the regularization matrix
    #g = get_reg_matrix(indicator_matrix, connected_reg, unconeected_reg)
    g = get_scale_reg_matrix(input_data_dict, num_targets, num_views)
    
    u = matrix_fact(indicator_matrix, g, alpha, max_iter, num_targets, num_views)
    nmf_results = get_concensus_matrix(u, num_targets, num_views)
    print(nmf_results)

if __name__ == "__main__":
    main()