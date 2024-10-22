import numpy as np
import struct
import os

class Data_Gene():
    def __init__(self, n_views, n_target):
        self.n = n_views
        self.q = n_target
        self.data_source = []
        self.data_source_matrix = [[self.q, self.n]]
        self.dist_matrix = [[self.q, self.n]]
    
    # Uniform distribution data generation function.
    # Lower bound of the distribution shouold be set to bound_low.
    # Upper bound of the distribution shouold be set to bound_up.
    def source_Gene(self, bound_low, bound_up):
        self.data_source = np.random.randint(bound_low,bound_up, size=(self.q, self.n))
        self.data_source_matrix.append(self.data_source.tolist())
        
    # Poisson distribution data generation function.
    # Lambda of the distribution shouold be set to l.
    def source_Gene_poi(self, l):
        self.data_source = np.random.poisson(lam=(l), size=(self.q, self.n))
        self.data_source_matrix.append(self.data_source.tolist())
        
    # Function to generate the distance matrix of input data.
    def distMatrix_Gene(self):
        for i in range(self.n):
            for j in range(i+1, self.n):
                local_dist_matrix = []
                for x in range(self.q):
                    dist_col = []
                    for y in range(self.q):
                        dist = abs(self.data_source[x,i] - self.data_source[y,j])
                        dist_col.append(dist)
                    local_dist_matrix.append(dist_col) 
                self.dist_matrix.append(local_dist_matrix)
          
    # Function to save the original data to user choose folder
    # filePath will be the relative path for savomh the original data to
    # Please un/comment corresponding data folder for the testing purpose.
    # 'original_data/': 5 uniform distribution of data sources with scale 0 to 100
    # 'uniform_1/': 4 uniform distribution of data sources with scale 0 to 100
    #               and 1 uniform distribution of data sources with scale 0 to 10
    # 'uniform_2/': 3 uniform distribution of data sources with scale 0 to 100
    #               and 2 uniform distribution of data sources with scale 0 to 10
    # 'poisson_1/': 4 uniform distribution of data sources with scale 0 to 100
    #               and 1 poisson distribution of data sources with mean 50
    def saveDatatoFile(self, index):
        filePath = 'original_data/'
        #filePath = 'uniform_1/'
        #filePath = 'uniform_2/'
        #filePath = 'poisson_1/'
        fileName = str(self.n) + 'D' + str(self.q) + '-' + str(index) + 'Original_Data' + '.dat'
        completeFileName = os.path.join(filePath, fileName)
        file = open(completeFileName, "w")
        for ele in self.data_source_matrix[0]:
            file.write(str(ele) + ' ')
        file.write('\n')
        file.write('\n')
        for ls in self.data_source_matrix[1]:
            for ele in ls:
                file.write(str(ele)+ ' ')
            file.write('\n')
        file.close()
        
    # Function to save the distance data to user choose folder
    # filePath will be the relative path for savomh the distance data to
    # Please un/comment corresponding data folder for the testing purpose.
    # 'dist_data/': 5 uniform distribution of data sources with scale 0 to 100
    # 'uniform_dist_1/': 4 uniform distribution of data sources with scale 0 to 100
    #                    and 1 uniform distribution of data sources with scale 0 to 10
    # 'uniform_dist_2/': 3 uniform distribution of data sources with scale 0 to 100
    #                    and 2 uniform distribution of data sources with scale 0 to 10
    # 'poisson_dist_1/': 4 uniform distribution of data sources with scale 0 to 100
    #                    and 1 poisson distribution of data sources with mean 50
    def saveDistoFile(self, index):
        filePath = 'dist_data/'
        #filePath = 'uniform_dist_1/'
        #filePath = 'uniform_dist_2/'
        #filePath = 'poisson_dist_1/'
        fileName = str(self.n) + 'D' + str(self.q) + '-' + str(index) + '.dat'
        completeFileName = os.path.join(filePath, fileName)
        file = open(completeFileName, "w")
        for ele in self.dist_matrix[0]:
            file.write(str(ele) + ' ')
        file.write('\n')
        file.write('\n')
        for ls in self.dist_matrix[1:]:
            for lls in ls:
                for ele in lls:
                    file.write(str(ele)+ ' ')  
                file.write('\n')
            file.write('\n')
        file.close()

# Main function to run the data generation function
# In the 2nd scale data generation in the later part of the function,
# a user should un/comment corresponding data generation functions.
# Here are the parameters that a user could change for testing purpose
# num_dim: number of sensors/stages for the data sources
# num_tar: Upper number of targets for the data sources
# dim_low_bound: Lower bound for number of sensors/stages for gneration of data sources
# dim_up_bound: Upper bound for number of sensors/stages for gneration of data sources
# target_low_bound: Lower bound for number of targets for gneration of data sources
# target_up_bound: Upper bound for number of targets for gneration of data sources
# bound_low: Lower bound for the uniform distribution used in gneration of data sources
# bound_up: Upper bound for the uniform distribution used in gneration of data sources
# num_generated_total: total number of data sources will be generated
# num_generated_scale_1: number of 1st scale data sources will be generated
# num_generated_scale_2: number of 2nd scale data sources will be generated
# l: lambda for the Poisson distribution used in gneration of data sources
def main():
    num_dim = 10
    num_tar = 20
    
    dim_low_bound = 3
    dim_up_bound = num_dim+1
    
    target_low_bound = 3
    target_up_bound = num_tar+1
    
    bound_low = 0
    bound_up = 100
    num_generated_total = 5
    num_generated_scale_1 = 4
    num_generated_scale_2 = num_generated_total - num_generated_scale_1
    
    for i in range(num_generated_scale_1):
        for j in range(dim_low_bound, dim_up_bound):
            for k in range(target_low_bound, target_up_bound):
                source = Data_Gene(j, k)
                source.source_Gene(bound_low,bound_up)
                source.distMatrix_Gene()
                source.saveDatatoFile(i+1)
                source.saveDistoFile(i+1)
    
    # a user should un/comment corresponding data generation functions.
    l = 50
    bound_low = 0
    bound_up = 10
    for i in range(num_generated_scale_2):
        for j in range(dim_low_bound, dim_up_bound):
            for k in range(target_low_bound, target_up_bound):
                source = Data_Gene(j, k)
                source.source_Gene_poi(l)
                #source.source_Gene(bound_low,bound_up)
                source.distMatrix_Gene()
                source.saveDatatoFile(num_generated_scale_1+1+i)
                source.saveDistoFile(num_generated_scale_1+1+i)
        
        
if __name__ == "__main__":
    main()
