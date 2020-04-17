#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 17:02:19 2020

@author: yiye
"""
import numpy as np
from numpy.linalg import multi_dot, inv, norm, eig
from scipy.sparse.linalg import cg
from time import time


# Timer
def tic():
    global t_start
    t_start = time()

def toc():
    if 't_start' in globals():
        print( 'Execution time is ' + str(time() - t_start) + ' seconds.')
    else:
        print( 'Start time is not set')


def power_iteration(A, num_simulations: int, tol = 0.1):
    """Power method to estimate the largest eigenvalue of positive semi-definite matrix.
    The following codes are from https://en.wikipedia.org/wiki/Power_iteration.
    """
    b_k = np.random.rand(A.shape[1])
    pho_k0 = np.inf
    for nbiter in range(num_simulations):      
        b_k1 = np.dot(A, b_k) # calculate the matrix-by-vector product Ab
        pho_k = np.dot(b_k1, b_k) # calculate the maximal eigenvalue       
        b_k1_norm = np.linalg.norm(b_k1) # calculate the norm       
        b_k = b_k1 / b_k1_norm # re normalize the vector
        if np.abs(pho_k0 - pho_k) < tol: 
            print("power iteration:")
            print("The tolerance {:2.3f} has been reached. The number of iterations is: {:}".format(tol, nbiter+1))
            break
        pho_k0 = pho_k
    return pho_k


# Helper functions.
# Conditioner
def plus_lda(K, lda):
    size = K.shape[0]
    if K.dtype is not type(lda):
        K = K.astype(type(lda))
    for i in range(size):
        K[i,i] += lda
    return K


# Multiple rhs conjugate gradient method
def mCG(A, B, tol):
    return np.array([cg(A = A,b = B[:,j],tol=tol)[0] for j in range(B.shape[1])]).T


# Product rule
def K_pd(K1, K2):
    assert K1.shape == K2.shape
    return np.multiply(K1,K2)


# Construct block toeplitz symmetric matrix from its first block row
def btoep(M):
    [m,n] = M.shape 
    K = np.zeros((n,n), dtype = np.float32)
    n = int(n/m) #m: block size; n: block number    
    K[:m,:] = M
    for l in np.arange(n-1)+1:
        K[l*m:(l+1)*m,:][:,l*m:] = M[:,:(n-l)*m]
    for i in range(n*m):
        for j in np.arange(i):
            K[i,j] = K[j,i]
    return K


# Gram matrices.
def k_rbf(t1, t2, gamma):
    """Gram matrix of Gaussian kernel"""
    n1 = len(t1)
    n2 = len(t2)
    K = np.zeros((n1,n2), dtype = np.float32)
    for i in range(n1):
        for j in range(n2):
            K[i,j] = np.exp(-gamma*(t1[i]-t2[j])**2)
    return K


def k_G(L, S1 = None, S2 = None):
    """Gram matrix of graph kernel"""
    w, v = eig(L)
    r = w
    r[r > 1e-05] = 1/r[r > 1e-05]
    K = multi_dot([v,np.diag(r),v.T])
    
    if S1 is not None:
        return K[S1,:][:,S2]
    else:
        return K
    
    
def k_T(t1, t2, tau, gamma, c):
    """Gram matrix of periodic kernel"""
    n1 = len(t1)
    n2 = len(t2)
    K = np.zeros((n1,n2), dtype = np.float32)
    for i in range(n1):
        for j in range(n2):
            K[i,j] = np.exp(-gamma*(t1[i]-t2[j])**2)*(np.cos(2*np.pi*np.abs(t1[i]-t2[j])/tau)+c)/(1+c)
    return K


def k_GT(L, gamma, St1, St2, tau = None, c = None):
    """Gram matrix of periodic spatial-temporal kernel"""
    S1 = [x for (x,_) in St1]
    t1 = [y for (_,y) in St1]
    S2 = [x for (x,_) in St2]
    t2 = [y for (_,y) in St2]
    if tau is not None:
        return K_pd(k_T(t1, t2, tau, gamma, c), k_G(L, S1, S2))
    else:
        return K_pd(k_rbf(t1, t2, gamma), k_G(L, S1, S2))


# Reconstruction errors.
def FLin0(i, I, Sigma_hat):
    S = [x for x in range(Sigma_hat.shape[1]) if x != i and x not in I]
    Sigma_iS = Sigma_hat[i, S]
    Sigma_S = Sigma_hat[S,:][:,S]
    return Sigma_hat[i,i] - multi_dot([Sigma_iS,inv(Sigma_S),Sigma_iS.T])    


def FLinH(i, I, H, Sigma_hat, AlphaH, lda = 0.0, inv_method = 'cg', tol = 1e-4): 
    N = Sigma_hat.shape[0]
    S = [x for x in range(int(AlphaH.shape[1]/(H+1))) if x != i and x not in I]    
    St = []
    for l in np.arange(H+1):
        St += [j+(N*l) for j in S]
    Beta_iS = AlphaH[i,St]
    Alpha_S = AlphaH[St,:][:,St]       
    
    if inv_method == 'direct':
        Theta_i_lda =  multi_dot([Beta_iS, inv(plus_lda(Alpha_S, lda))])
    elif inv_method == 'cg':
        Theta_i_lda = cg(plus_lda(Alpha_S, lda),Beta_iS.T, tol=tol)[0]
    else:
        raise NameError('The inverse method is undefined.')       
    return Sigma_hat[i,i] -  multi_dot([Beta_iS, Theta_i_lda.T])


def FKer0(i, I, Sigma_hat, K, lda = 0.0, inv_method = 'direct', tol = None):
    S = [x for x in range(Sigma_hat.shape[1]) if x != i and x not in I]
    Sigma_iS = Sigma_hat[[i],:][:,S]   
    Sigma_S = Sigma_hat[S,:][:,S]
    
    K_lda = plus_lda(K[S,:][:,S], lda)
    if inv_method == 'direct':
        Theta_i_lda =  multi_dot([K[i, S], inv(K_lda)])
        Theta_i_lda = np.squeeze(np.array(Theta_i_lda))
    elif inv_method == 'cg':
        Theta_i_lda = cg(K_lda, K[i, S].T, tol=tol)[0]
    else:
        raise NameError('The inverse method is undefined.')
    return Sigma_hat[i,i] - 2*multi_dot([Sigma_iS,Theta_i_lda.T]) + multi_dot([Theta_i_lda,Sigma_S,Theta_i_lda.T])


def FKerH(i, I, H, Sigma_hat, AlphaH, KH, lda = 0.0, inv_method = 'direct', tol = None):
    N = Sigma_hat.shape[0]
    S = [x for x in range(int(Sigma_hat.shape[1])) if x != i and x not in I]    
    St = []
    for l in np.arange(H+1):
        St += [j+(N*l) for j in S]
    KH_iS = KH[i,St]
    Beta_iS = AlphaH[i,St]
    KH_S = KH[St,:][:,St]
    Alpha_S = AlphaH[St,:][:,St]
    
    KH_S_lda = plus_lda(KH_S, lda)    
    if inv_method == 'direct':
        Theta_i_lda =  multi_dot([KH_iS, inv(KH_S_lda)])
        Theta_i_lda = np.squeeze(np.array(Theta_i_lda))
    elif inv_method == 'cg':
        Theta_i_lda = cg(KH_S_lda,KH_iS.T, tol=tol)[0]
    else:
        raise NameError('The inverse method is undefined.')
    return Sigma_hat[i,i] -  2*multi_dot([Beta_iS,Theta_i_lda]) + multi_dot([Theta_i_lda.T,Alpha_S,Theta_i_lda])


def alg1(Sigma_hat, P): 
    """ Algorithm 1: Linear regression, H = 0   
    Input:
        Sigma_hat: sample covariance matrix of stationary network time series.
        P: number of sensors to be selected.      
    Output: 
        I: row (=column) indeices of selected sensors in Sigma_hat.
    """
    I = []
    for q in range(P):
        Ic = [i for i in range(Sigma_hat.shape[1]) if i not in I]
        I += [Ic[np.argmin(np.array([FLin0(i, I, Sigma_hat) for i in Ic]))]]
    return I


def alg2(Sigma_hat, AlphaH, P, H, lda, inv_method = 'cg', tol = 1e-4): 
    """ Algorithm 2/4: Linear (ridge) regression, H > 0
    Input:
        Sigma_hat: sample covariance matrix of stationary network time series.
        AlphaH: sample covariance matrix of x^H_{N,t} in the paper.
        P: number of sensors to be selected.      
        H: time lag.
        lda: lambda, 0  , the algorithm implements linear regression;
                     >0 , implements linear ridge regression.
        inv_method: matrix inversion method, 
                    either direction inversion, 'direct', 
                    or conjugate gradient descent, 'cg'.
        tol: convergence tolerance of conjugate gradient descent algorithm.
    Output: 
        I: row (=column) indeices of selected sensors in Sigma_hat.
    """
    I = []
    for q in range(P):
        if 100*q/P % 20 < 5:
            print("The percentage done is: %3.2f"% (q/P)) 
        Ic = [i for i in range(Sigma_hat.shape[1]) if i not in I]
        I += [Ic[np.argmin(np.array([FLinH(i, I, H, Sigma_hat, AlphaH, lda, inv_method = 'cg', tol = tol) for i in Ic]))]]
    return I


def alg3(Sigma_hat, K, P, lda, inv_method = 'direct', tol = None): 
    """ Algorithm 3: Kernel ridge regression, H = 0.
    Input:
        Sigma_hat: sample covariance matrix of stationary network time series.
        K: Gram matrix of vertex set.
        P: number of sensors to be selected.   
        lda: lambda.
        inv_method: matrix inversion method, 
                    either direction inversion, 'direct', 
                    or conjugate gradient descent, 'cg'.
        tol: convergence tolerance of conjugate gradient descent algorithm.
    Output: 
        I: row (=column) indeices of selected sensors in Sigma_hat.
    """        
    I = []
    for _ in range(P):
        Ic = [i for i in range(Sigma_hat.shape[1]) if i not in I]
        I += [Ic[np.argmin(np.array([FKer0(i, I, Sigma_hat, K, lda, inv_method, tol) for i in Ic]))]]
    return I


def alg4(Sigma_hat, KH, AlphaH, P, H, lda, inv_method = 'cg', tol = 1e-4):
    """ Algorithm 4: Kernel ridge regression, H > 0.
    Input:
        Sigma_hat: sample covariance matrix of stationary network time series.
        KH: Gram matrix of vertex set \times (t, t-1, ..., t - H).
        AlphaH: sample covariance matrix of x^H_{N,t} in the paper.
        P: number of sensors to be selected.  
        H: time lag.
        lda: lambda.
        inv_method: matrix inversion method, 
                    either direction inversion, 'direct', 
                    or conjugate gradient descent, 'cg'.
        tol: convergence tolerance of conjugate gradient descent algorithm.
    Output: 
        I: row (=column) indeices of selected sensors in Sigma_hat.
    """       
    I = []
    for q in range(P):
        if 100*q/P % 20 < 5:
            print("The percentage done is: %3.2f"% (q/P)) 
        Ic = [i for i in range(Sigma_hat.shape[1]) if i not in I]
        I += [Ic[np.argmin(np.array([FKerH(i, I, H, Sigma_hat, AlphaH, KH, lda, inv_method, tol) for i in Ic]))]]
    return I
    

# Reconstruction functions.
def ReLin0(I, Sigma_hat, X, Y, trend = 0.0, sd = 1.0):
    """ Reconstruction function of linear regression, H = 0.
    Input:
        I: turned-off set, a list of indices of selected sensors,
           which correspond to the row (=column) indices of Sigma_hat.        
        Sigma_hat: sample covariance matrix of stationary network time series.
        X: observations of turned-on sensors, columns represent sensors.
        Y: ground truth of turned-ooff sensors, columns represent sensors.
        trend (optional): trend of turned-off sensors.
        sd (optional): standard deviation of turned-off sensors.
    Output:
        X_I_hat: predictions of turned-off sensors, with or without trend and/or standard deviation back.
        err_mse: MSE of prediction without trend and/or standard deviation back.
    """
    Ic = list(set(range(Sigma_hat.shape[0])) - set(I))
    Sigma_IIc = Sigma_hat[I,:][:,Ic]
    Sigma_Ic = Sigma_hat[Ic,:][:,Ic]
    Theta_hat = multi_dot([Sigma_IIc,inv(Sigma_Ic)])
    
    X_I_hat = multi_dot([Theta_hat,X.T])
    X_I_hat = X_I_hat.T
    err_mse = norm(np.multiply(Y - X_I_hat, sd))**2/Y.shape[0]

    X_I_hat = np.multiply(X_I_hat, sd) # Multiply the std back
    X_I_hat += trend # Add the trend back
    return [X_I_hat, err_mse]


def ReLinH(I, H, AlphaH, X, Y, lda = 0.0, trend = 0.0, sd = 1.0, inv_method = 'cg', tol = 1e-4):
    """ Reconstruction function of linear (ridge) regression, H > 0.
    Input:
        I: turned-off set, a list of indices of selected sensors,
           which correspond to the row (=column) indices of Sigma_hat.        
        H: time lag.
        AlphaH: sample covariance matrix of x^H_{N,t} in the paper.
        X: observations of turned-on sensors, columns represent sensors.
        Y: ground truth of turned-ooff sensors, columns represent sensors.
        lda: lambda.
        trend (optional): trend of turned-off sensors.
        sd (optional): standard deviation of turned-off sensors.
        inv_method (optional): matrix inversion method, 
                               either direction inversion, 'direct', 
                               or conjugate gradient descent, 'cg'.
        tol (optional): convergence tolerance of conjugate gradient descent algorithm.
    Output:
        X_I_hat: predictions of turned-off sensors, with or without trend and/or standard deviation back.
        err_mse: MSE of prediction without trend and/or standard deviation back.
    """
    N = int(X.shape[1]/(H+1)) + Y.shape[1]
    Ic = list(set(range(N)) - set(I))      
    St = []
    for l in np.arange(H+1):
        St += [j+(N*l) for j in Ic]
    Beta_IIc = AlphaH[I,:][:,St]
    Alpha_Ic = AlphaH[St,:][:,St]
    
    Alpha_Ic_lda = plus_lda(Alpha_Ic, lda)  
    if inv_method == 'direct':
        Theta_lda =  multi_dot([Beta_IIc, inv(Alpha_Ic_lda)])
    elif inv_method == 'cg':
        Theta_lda = mCG(Alpha_Ic_lda, Beta_IIc.T, tol = tol)
        Theta_lda = Theta_lda.T
    else:
        raise NameError('The inverse method is undefined.')
    
    X_I_hat = multi_dot([Theta_lda,X.T])
    X_I_hat = X_I_hat.T
    err_mse = norm(np.multiply(Y - X_I_hat, sd))**2/Y.shape[0]

    X_I_hat = np.multiply(X_I_hat, sd) # Multiply the std back
    X_I_hat += trend # Add the trend back
    return [X_I_hat, err_mse]


def ReKer0(I, K, X, Y, lda = 0.0, trend = 0.0, sd = 1.0, inv_method = 'direct', tol = 1e-4):
    """ Reconstruction function of kernel ridge regression, H = 0.
    Input:
        I: turned-off set, a list of indices of selected sensors,
           which correspond to the row (=column) indices of Sigma_hat.    
        K: Gram matrix of vertex set.
        X: observations of turned-on sensors, columns represent sensors.
        Y: ground truth of turned-ooff sensors, columns represent sensors.
        lda: lambda.
        trend (optional): trend of turned-off sensors.
        sd (optional): standard deviation of turned-off sensors.
        inv_method (optional): matrix inversion method, 
                               either direction inversion, 'direct', 
                               or conjugate gradient descent, 'cg'.
        tol (optional): convergence tolerance of conjugate gradient descent algorithm.
    Output:
        X_I_hat: predictions of turned-off sensors, with or without trend and/or standard deviation back.
        err_mse: MSE of prediction without trend and/or standard deviation back.
    """
    N = X.shape[1] + Y.shape[1]
    Ic = list(set(range(N)) - set(I)) 
    if inv_method == 'direct':
        Theta_lda =  multi_dot([K[I,:][:,Ic], inv(plus_lda(K[Ic,:][:,Ic], lda))])
    elif inv_method == 'cg':
        Theta_lda = mCG(plus_lda(K[Ic,:][:,Ic], lda), K[I,:][:,Ic].T, tol = tol)
        Theta_lda = Theta_lda.T
    else:
        raise NameError('The inverse method is undefined.')
    
    X_I_hat = multi_dot([Theta_lda,X.T])
    X_I_hat = X_I_hat.T
    err_mse = norm(np.multiply(Y - X_I_hat, sd))**2/Y.shape[0]

    X_I_hat = np.multiply(X_I_hat, sd) # Multiply the std back
    X_I_hat += trend # Add the trend back
    return [X_I_hat, err_mse]


def ReKerH(I, H, KH, X, Y, lda = 0.0, trend = 0.0, sd = 1.0, inv_method = 'cg', tol = 1e-4):
    """ Reconstruction function of kernel ridge regression, H > 0.
    Input:
        I: turned-off set, a list of indices of selected sensors,
           which correspond to the row (=column) indices of Sigma_hat.  
        H: time lag.
        KH: Gram matrix of vertex set \times (t, t-1, ..., t - H).
        X: observations of turned-on sensors, columns represent sensors.
        Y: ground truth of turned-ooff sensors, columns represent sensors.
        lda: lambda.
        trend (optional): trend of turned-off sensors.
        sd (optional): standard deviation of turned-off sensors.
        inv_method (optional): matrix inversion method, 
                               either direction inversion, 'direct', 
                               or conjugate gradient descent, 'cg'.
        tol (optional): convergence tolerance of conjugate gradient descent algorithm.
    Output:
        X_I_hat: predictions of turned-off sensors, with or without trend and/or standard deviation back.
        err_mse: MSE of prediction without trend and/or standard deviation back.
    """

    N = int(X.shape[1]/(H+1)) + Y.shape[1]
    Ic = list(set(range(N)) - set(I))   
    St = []
    for l in np.arange(H+1):
        St += [j+(N*l) for j in Ic]
    KH_IIc = KH[I,:][:,St]
    KH_Ic = KH[St,:][:,St]
    
    KH_Ic_lda = plus_lda(KH_Ic, lda)  
    if inv_method == 'direct':
        Theta_lda =  multi_dot([KH_IIc, inv(KH_Ic_lda)])
    elif inv_method == 'cg':
        Theta_lda = mCG(KH_Ic_lda, KH_IIc.T, tol = tol)
        Theta_lda = Theta_lda.T
    else:
        raise NameError('The inverse method is undefined.')
    
    X_I_hat = multi_dot([Theta_lda,X.T])
    X_I_hat = X_I_hat.T
    err_mse = norm(np.multiply(Y - X_I_hat, sd))**2/Y.shape[0]

    X_I_hat = np.multiply(X_I_hat, sd) # Multiply the std back
    X_I_hat += trend # Add the trend back
    return [X_I_hat, err_mse]    

