# Code Repository for Computational Framework for Target Tracking Information Fusion Problems
This code repository is for the Computational Framework for Target Tracking Information Fusion Problems by T. Yang, J. Liu, T. Faiz, C. Vogiatzis, and Md. Noor-E-Alam. All the data and code used in the paper will be provided in both Jupiter Notebook and Python format. Linear programming presented in the paper is solved by Gurobi. Users may be able to reproduce all the results presented in the paper and perform further testing by following the instructions. 

Link to the paper: 

The Jupiter Notebook file for all methods described in the paper is available in the Jupiter_Code subfolder of this repository. 

*Jupiter Notebook version used:*
- Jupiter Notebook version 6.5.2 in Anaconda Distribution version 2.3.1

The Python file for all methods described in the paper is available in the Python_Code subfolder of this repository. 

*Python version used:*
- Python version 3.12.0

*Python packages required to run the code:*
- numpy
- pandas
- struct
- os
- gurobipy
- time
- scipy
- matplotlib
- xlsxwriter
- itertools

*Gurobi version used*
- Gurobi version 9.0.2

**Before trying to run any of the files in your environment please make sure that all the paths point to the correct folders, paste the data under the corresponding relative path, and go over the function comments in the code.**

## To reproduce Data Generation
* Run: Data Generation.py/Data_Generation.ipynb
* Note:
  * Dimensions, Distributions, and Scales for the generated data can be controlled in the code file.
  * All the uniform distributions should be scaled non-negative.
  * Users should prepare the relative paths before running the Data Generation files. 
  * Generated Data Files will be saved in the user-chosen folder in the saveDatatoFile and saveDistoFile functions.

## To reproduce Single-source Methods Results alone
* Run: single_source_algo.py/single_source_algo.ipynb
* Note:
  * Dimensions, Distributions, and Scales for the input data can be controlled in the code file.
  * Under medium dimensions (10 sensors/stages, 20 targets), LP will face computational difficulties.
  * Under large dimensions (over 30 sensors/stages, 30 targets), LP will not be able to solve in a reasonable time, and RMSRA will take a longer time to solve. 
  * User should un/comment the print statement to review the results for corresponding methods.

 ## To reproduce Multi-source Methods Results with Single-source Methods
* Run: test_algos.py/test_algos.ipynb
* Note:
  * Dimensions, Distributions, and Scales for the input data can be controlled in the code file.
  * For single-source tests, please refer to the previous instruction on computational difficulties among LP and RMSRA.
  * For multi-source tests, all the input information should match with the corresponding outputs of the single-source methods. 
  * User should un/comment the print statement to review the results for corresponding methods.
  
 ## To reproduce Noisy Situation Testing
* Run: noise_test.py/noise_test.ipynb
* Note:
  * Dimensions, Distributions, and Scales for the input data can be controlled in the code file.
  * For single-source noisy tests, users may select path-based or clique-based formulation besides the input data control
  * For multi-source tests, CBSNMF can work with multiple similarity matrices and regularization matrices.
  * User should un/comment the print statement to review the results for corresponding methods.
