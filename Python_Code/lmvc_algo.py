import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from scipy.optimize import linear_sum_assignment

class Data():
    def __init__(self, num_views, num_targets, phase_one_path):
        self.m = len(phase_one_path)
        self.n = num_views
        self.q = num_targets
        self.phase_one_path = phase_one_path
        self.ind_mats = {}
        self.blocks = {}

    def process(self):
        self.get_ind_mats()
        self.block_each_source()
        filter_blocks = []
        for i in range(self.n - 1):
            if (i == 0):
                filter_blocks.append(np.identity(self.q))
            j = i + 1
            true_mat, S_mat = self.find_S_mat(i, j)
            '''
            find initial Laplacian embedding matrix F and find corresponding v matrix
            '''
            norm_c_mat = true_mat
            norm_c_mat = norm_c_mat / norm_c_mat.sum(axis=0, keepdims=1)
            norm_c_mat = self.expand_block(norm_c_mat)  # normalize and expand
            F_init = self.F_mat_update(norm_c_mat)  # initialization of F matrix
            F_mat = F_init

            flag = 1
            ite = 0
            bound = 5
            while ite < bound:
                '''
                fix F and update U
                '''
                U = self.U_update(S_mat, F_mat, ite, scale=10 ** 6)

                '''
                fix U and update F
                '''
                F_mat = self.F_mat_update(U)
                ite += 1

            P_i = self.const_aux_mat(2, self.q, 0, 1)
            P_j = self.const_aux_mat(2, self.q, 1, 0)

            sub_block = np.dot(np.dot(P_i[0], U), P_j[0].T)

            row_ind, col_ind = linear_sum_assignment(-sub_block)
            sub_block_final = np.array([[0 for i in range(self.q)] for j in range(self.q)])
            for (i,j) in zip(row_ind,col_ind):
                sub_block_final[i][j] = 1

            filter_blocks.append(sub_block_final)
        consensus = np.block(filter_blocks)
        path = self.path_retrieve(consensus)
        return path

    def get_ind_mats(self):
        count = 0
        for i in range(self.m):
            ind_mat = [[0 for i in range(self.n * self.q)] for j in range(self.n * self.q)]
            path = self.phase_one_path[i]
            for p in path:
                p = [p[i] + self.q * i for i in range(self.n)]
                for i in p:
                    for j in p:
                        ind_mat[i][j] = 1
            self.ind_mats[count]=(np.array(ind_mat))
            count += 1

    # construct auxiliary matrix used in patitioning the matrix
    def const_aux_mat(self, n, q, i, j):
        '''
        :param n: num of views or the number of blocks each row
        :param q: num of targets or the size of blocks
        :param i: index for the row block
        :param j: index for the column block
        :return: constructed auxiliary matrix
        '''
        I_mat = np.identity(q)
        o_mat = np.zeros((q, q))
        P_i = np.block([I_mat if k == i else o_mat for k in range(n)])
        P_j = np.block([I_mat if k == j else o_mat for k in range(n)])
        return (P_i, P_j)

    def block_each_source(self):
        for k in range(self.m):
            ind_mat = self.ind_mats[k]
            tmp = np.array(ind_mat).shape
            block = [[0 for i in range(self.n)] for j in range(self.n)]
            for i in range(self.n):
                for j in range(self.n):
                    aux_mats = self.const_aux_mat(self.n, self.q, i, j)
                    block[i][j] = [np.dot(np.dot(aux_mats[0], ind_mat), aux_mats[1].T)]
            self.blocks[k] = np.array(block)

    # normalize U to set the value of each row to be 1
    def U_norm(self, U, q, threshold=10 ** (-5)):
        '''
        :param U: consensus matrix
        :param q: dimension of the matrix
        :param threshold: threshold for assigning 1 to the corresponding position
        :return: normalized U matrix with highest value for each row set to be 1
        '''
        s = np.sum(U, axis=0)
        for j in range(q):
            for i in range(q, 2 * q):
                if s[j] == 0:
                    U[i][j] = 0.0
                else:
                    U[i][j] = U[i][j] / s[j]

        for i in range(q):
            for j in range(q, 2 * q):
                U[i][j] = U[j][i]
        return U

    def S_norm(self, S_mat, bool_mat):
        '''
        :param S_mat: the S matrix that is ready to be optimized
        :param bool_mat: the bool matrix that indicates the occurence for each connection
        :param q: the number of targets (clusters)
        :return: the normalized S matrix
        '''
        row_sums = S_mat.sum(axis=1)
        for i in range(self.q):
            for j in range(self.q):
                if (bool_mat[i][j] == 0):
                    S_mat[i][j] = 0
                else:
                    S_mat[i][j] = S_mat[i][j] / row_sums[i]
        return S_mat

    def expand_block(self, mat):
        o_mat = np.zeros((self.q, self.q))
        expand_block = np.block([
            [o_mat, mat],
            [mat.T, o_mat]
        ])
        return expand_block

    def find_S_mat(self, i, j):
        true_mat = 0
        for k in range(self.m):
            true_mat += self.blocks[k][i][j][0].astype(float)
        bool_mat = np.array(true_mat, dtype=bool).astype(int)
        true_mat_comp = self.S_norm(true_mat, bool_mat)
        return true_mat, self.expand_block(true_mat_comp)

    def F_mat_update(self, mat):
        '''
        :param mat: probability of the true matrix
        :param q: dimension of the matrix
        :return: expanded blocks
        '''
        lap_mat = laplacian(mat)
        _, vecs = eigs(lap_mat, k=self.q, which='SR')  # find q smallest eigenvalues
        vecs = np.real(vecs)  # get real part of vecs
        return vecs

    # update U when S and F are fixed
    def U_update(self, S_mat, F_mat, ite, scale=1):
        '''
        :param S_mat: similarity matrix or affinity matrix for each source
        :param F_mat: embedded matrix
        :param q: dimension of the matrix
        :param ite: iteration times
        :param scale: the hyperparameter that could be as large as possible
        :return: updated consensus matrix U
        '''

        U = np.array([[0.0 for i in range(2 * self.q)] for j in range(2 * self.q)])

        U_old = S_mat
        def v_mat(F_mat, q):
            '''
            :param F_mat: embedded matrix
            :param q: dimension of the matrix
            :return: V matrix representing the difference between the row vector (V is symmetric)
            '''
            v = np.array([[0.0 for i in range(2 * q)] for j in range(2 * q)])
            for i in range(2 * q):
                for j in range(2 * q):
                    v_ij = F_mat[i] - F_mat[j]
                    v[i][j] = np.dot(v_ij, v_ij)
            return v

        V = v_mat(F_mat, self.q)
        sum_V = V.sum(axis=1)
        V = V / sum_V[:, np.newaxis]

        eta = np.array([0.0 for i in range(2 * self.q)])

        lam = self.q * scale

        for i in range(self.q):
            sum = 0.0
            for j in range(self.q, 2 * self.q):
                sum += S_mat[i][j] - 1 / 2 * lam * V[i][j]
            eta[i] = (1 - sum) / self.q

        for i in range(self.q, 2 * self.q):
            sum = 0.0
            for j in range(0, self.q):
                sum += S_mat[i][j] - 1 / 2 * lam * V[i][j]
            eta[i] = (1 - sum) / self.q

        for i in range(self.q):
            for j in range(self.q, 2 * self.q):
                U[i][j] = max(0, S_mat[i][j] - 1 / 2 * lam * V[i][j] + eta[i])

        for i in range(self.q, 2 * self.q):
            for j in range(self.q):
                U[i][j] = max(0, S_mat[i][j] - 1 / 2 * lam * V[i][j] + eta[i])
        U = self.U_norm(U, self.q)

        row_ind, col_ind = linear_sum_assignment(-U)
        U_final = np.array([[0.0 for i in range(2*self.q)] for j in range(2*self.q)])
        for (i, j) in zip(row_ind, col_ind):
            U_final[i][j] = 1.0
            
        return U_final

    # check if we need the loop or not
    def checkflag(self, U, ite):
        '''
        :param U: normalized consensus matrix U
        :param q: the number of targets (clusters)
        :param ite: iterate times
        :return: flag which indicates whether iteration should be continued or stopped
        '''
        flag = 0

        # check if sum(U_ij) = 1 and only one nonzero element for each column
        for j in range(2 * self.q):
           sum = 0
           for i in range(2 * self.q):
               sum += (U[i][j] == 1)
           flag += (sum != 1)  # if sum!=1 loop again

        # check symmetric constraint
        P_i = self.const_aux_mat(2, self.q, 0, 1)
        P_j = self.const_aux_mat(2, self.q, 1, 0)

        B1 = np.dot(np.dot(P_i[0], U), P_j[0].T)
        B2 = np.dot(np.dot(P_j[0], U), P_i[0].T)
        flag += np.sum((B1 != B2.T))

        return flag, B1, ite

    # retrieve path from H_star
    def path_retrieve(self, H_star):
        '''
        :param H_star: the optimized indicator matrix that indicates the path (size: q*nq)
        :param n: the number of sensors / the number of columns for each matrix
        :param q: the number of targets / the number of rows for each matrix
        :return: the path indicated by H_star
        '''
        path = [[i] for i in range(self.q)]
        for j in range(self.n - 1):
            H_end = H_star[:, (j + 1) * self.q:(j + 2) * self.q]
            for s in range(self.q):
                for e in range(self.q):
                    if H_end[s][e] == 1:
                        for m in range(self.q):
                            if path[m][j] == s:
                                path[m].append(e)
                                break
        return path