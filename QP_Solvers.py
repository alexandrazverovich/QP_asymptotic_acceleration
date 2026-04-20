import numpy as np
import numpy.linalg as la

## Tool function: computation of inverse vector.
# Input: vector. 
# Output: Inverse of the vector (1/vector for non-zero values; 0 otherwise).
def g_inv_vec(vec): 
    N = len(vec) ; ginv = np.zeros(N, dtype=float)
    for i in range(N):
        if vec[i] != 0.0 : ginv[i]=1/vec[i] 
        else : ginv[i] = 0.0
    return ginv  




######################
## Coordinate descent on the relaxed map R with H-Coordinate + information related to the acceleration term 
## (a faster solver without the additional recordings is provided below; see R_HCoo_Solver).
# Inputs: number of iterations, matrix Q, vector c, constant term D0.
# Outputs: candidate solution x, record of D map, number of iterations taken, terms required to compute acceleration term (xQx, vQx, vQv).
def R_HCoo_Solver_All(n_iter,   # maximum number of iterations
                      Q_mat,    # matrix Q
                      c_vec,    # vector c 
                      D0):      # value of D(0)
    
    N = len(c_vec)
    sqrt_diag_Q_inv = g_inv_vec(np.sqrt(np.abs(np.diag(Q_mat))))
    
    ## Record
    R_rec = np.zeros(n_iter+1, dtype=float) ## record of R(v)
    R_rec[0] = D0
    
    ## Initialisation
    u = np.argmax(np.abs(c_vec * sqrt_diag_Q_inv ))
    sign_c = np.sign(c_vec[u])
    
    x_coo = np.zeros(N, dtype=float)
    x_coo[u] = sign_c
    Qx_vec = sign_c * Q_mat[:,u] 
    xQx = Q_mat[u,u] 
    cx = sign_c * c_vec[u] 
    sx = cx/xQx
    res = sx * Qx_vec - c_vec
    R_rec[1] = D0 - (cx**2)/xQx
    
    xQx_rec = np.zeros(n_iter, dtype=float)
    vQx_rec = np.zeros(n_iter, dtype=float)
    vQv_rec = np.zeros(n_iter, dtype=float)
    
    ## Iterate
    for q in range(1,n_iter):
        
        v = np.argmax(np.abs(res * sqrt_diag_Q_inv))
        
        if res[v]==0 : 
            #print('zero grad')
            q = q-1 ; break # minimum is reached
            
        Up_XV =  c_vec[v] * xQx  - cx * Qx_vec[v]  
        Up_VX =  cx * Q_mat[v,v]  - c_vec[v] * Qx_vec[v]  
        
        ## Rec
        xQx_rec[q-1] = xQx
        vQx_rec[q-1] = Qx_vec[v]
        vQv_rec[q-1] = Q_mat[v,v]
      
        ## Update 
        rho =  Up_XV / Up_VX
        xQx += (rho**2) * Q_mat[v,v] + 2 * rho * Qx_vec[v]
        Qx_vec += rho * Q_mat[:,v]  
        cx += rho * c_vec[v] 
        sx = cx/xQx
        res = sx * Qx_vec - c_vec

        R_rec[q+1] = D0 - (cx**2)/xQx
        if R_rec[q+1] >= R_rec[q] : 
            #print('cost increase')
            q = q-1 ; break
        if R_rec[q+1] < 0 : 
            #print('negative cost')
            q = q-1 ; break
            
        ## Update x^(k) after check in case of break
        x_coo[v] += rho     
    
    output = {'x': sx * x_coo,           # final iterate x
              'D': R_rec[:(q+1)],        # evolution of the cost D
              'q': q,                    # index of the final iteration
              'xQx': xQx_rec[:(q)],      # evolution of the quantity xQx
              'vQx': vQx_rec[:(q)],      # evolution of the quantity vQx
              'vQv': vQv_rec[:(q)] }     # evolution of the quantity vQv
    
    return output



## Coordinate descent on the relaxed map R with H-Coordinate.
# Inputs: number of iterations, matrix Q, vector c, constant term D0.
# Outputs: candidate solution x, record of D map, number of iterations taken.
def R_HCoo_Solver(n_iter,   # maximum number of iterations
                  Q_mat,    # matrix Q
                  c_vec,    # vector c 
                  D0):      # value of D(0)
    
    N = len(c_vec)
    sqrt_diag_Q_inv = g_inv_vec(np.sqrt(np.abs(np.diag(Q_mat))))
    
    ## Record
    R_rec = np.zeros(n_iter+1, dtype=float) ## record of R(v)
    R_rec[0] = D0
    
    ## Initialisation
    u = np.argmax(np.abs(c_vec * sqrt_diag_Q_inv ))
    sign_c = np.sign(c_vec[u])
    
    x_coo = np.zeros(N, dtype=float)
    x_coo[u] = sign_c
    Qx_vec = sign_c * Q_mat[:,u] 
    xQx = Q_mat[u,u] 
    cx = sign_c * c_vec[u] 
    sx = cx/xQx
    res = sx * Qx_vec - c_vec
    R_rec[1] = D0 - (cx**2)/xQx
    
    ## Iterate
    for q in range(1,n_iter): 
        
        v = np.argmax(np.abs(res * sqrt_diag_Q_inv))
        
        if res[v]==0 : 
            #print('zero grad')
            q = q-1 ; break # minimum is reached
            
        Up_XV =  c_vec[v] * xQx  - cx * Qx_vec[v]  
        Up_VX =  cx * Q_mat[v,v]  - c_vec[v] * Qx_vec[v]  
      
        rho =  Up_XV / Up_VX
        xQx += (rho**2) * Q_mat[v,v] + 2 * rho * Qx_vec[v]
        Qx_vec += rho * Q_mat[:,v]  
        cx += rho * c_vec[v] 
        sx = cx/xQx
        res = sx * Qx_vec - c_vec

        R_rec[q+1] = D0 - (cx**2)/xQx
        if R_rec[q+1] >= R_rec[q] : 
            #print('cost increase')
            q = q-1 ; break
        if R_rec[q+1] < 0 : 
            #print('negative cost')
            q = q-1 ; break
            
        ## Update x^(k) after check in case of break
        x_coo[v] += rho     
    
    output = {'x': sx * x_coo,      # final iterate x
              'D': R_rec[:(q+1)],   # evolution of the cost D
              'q': q }              # index of the final iteration
    
    return output



## Coordinate descent on the relaxed map R with BI-coordinate. 
# Inputs: number of iterations, matrix Q, vector c, constant term D0.
# Outputs: candidate solution x, record of D map, number of iterations taken. 
def R_BICoo_Solver(n_iter,   # maximum number of iterations
                   Q_mat,    # matrix Q
                   c_vec,    # vector c 
                   D0):      # value of D(0)
    
    N = len(c_vec)
    diag_Q = np.diag(Q_mat)
    sqrt_diag_Q_inv = g_inv_vec(np.sqrt(np.abs(diag_Q)))
    
    ## Record
    R_rec = np.zeros(n_iter+1, dtype=float) ## record of R(v)
    R_rec[0] = D0
    
    ## Initialisation
    u = np.argmax(np.abs(c_vec * sqrt_diag_Q_inv ))
    sign_c = np.sign(c_vec[u])
    
    x_coo = np.zeros(N, dtype=float)
    x_coo[u] = sign_c
    Qx_vec = sign_c * Q_mat[:,u] 
    xQx = Q_mat[u,u] 
    cx = sign_c * c_vec[u] 
    sx = cx/xQx
    res = sx * Qx_vec - c_vec
    R_rec[1] = D0 - (cx**2)/xQx
    
    ## Iterate
    for q in range(1,n_iter): 
        
        V4_BI_selec = (diag_Q - (Qx_vec**2)/xQx)
        V4_BI_selec[u] = -1 
        u = np.argmax((res**2) / V4_BI_selec)
        
        if res[u]==0 : 
            #print('zero grad')
            q = q-1 ; break # minimum is reached
            
        Up_XV = ( c_vec[u] * xQx  - cx * Qx_vec[u] ) 
        Up_VX = ( cx * Q_mat[u,u]  - c_vec[u] * Qx_vec[u] )  
        
        r = Up_XV / Up_VX
        xQx += (r**2) * Q_mat[u,u] + 2 * r * Qx_vec[u]
        Qx_vec += r * Q_mat[:,u]  
        cx += r * c_vec[u] 
        sx = cx/xQx
        res = sx * Qx_vec - c_vec

        R_rec[q+1] = D0 - (cx**2)/xQx
        if R_rec[q+1] > R_rec[q] : 
            #print('cost increase')
            q = q-1 ; break 
        if R_rec[q+1] < 0 : 
            #print('negative cost')
            q = q-1 ; break
            
        ## Update x^(k) after check in case of break
        x_coo[u] += r     
    
    output = {'x': sx * x_coo,           # final iterate x
              'D': R_rec[:(q+1)],        # evolution of the cost D
              'q': q }                   # index of the final iteration
    
    return output




## Coordinate descent (Gauss-Seidel) on quadratic map D.
# Inputs: number of iterations, matrix Q, vector c, constant term D0.
# Outputs: candidate solution x, record of D map, record of R map, number of iterations taken. 
def D_GSL_Solver(n_iter,   # maximum number of iterations
                 Q_mat,    # matrix Q
                 c_vec,    # vector c 
                 D0):      # value of D(0)  
    
    N = len(c_vec)
    sqrt_diag_Q_inv = g_inv_vec(np.sqrt(np.abs(np.diag(Q_mat))))
    
    D_rec = np.zeros(n_iter+1, dtype=float) # record D(v)
    R_rec = np.zeros(n_iter+1, dtype=float) # record R(v)
    x_coo = np.zeros(N, dtype=float)        
    Qx_vec = np.zeros(N, dtype=float)
    xQx = 0
    cx = 0
    res = - c_vec
    D_rec[0] = D0
    R_rec[0] = D0
    
    for q in range(0,n_iter):
    
        ## Selection 
        v = np.argmax(np.abs(res * sqrt_diag_Q_inv))
        r = -res[v] / Q_mat[v,v]
        
        if res[v]==0 : 
            #print('zero grad')
            q = max(0,q-1) ; break # minimum is reached
        
        ## Update 
        xQx += (r**2) * Q_mat[v,v] + 2 * r * Qx_vec[v] 
        Qx_vec += r * Q_mat[:,v]  
        cx += r * c_vec[v] 
        res = Qx_vec - c_vec
        
        D_rec[q+1] = D0 + xQx - 2 * cx
        R_rec[q+1] = D0 - (cx**2) / xQx 
        if D_rec[q+1] >= D_rec[q] : 
            #print('cost increase')
            q = max(0,q-1) ; break
        if D_rec[q+1] < 0 : 
            #print('negative cost')
            q = max(0,q-1) ; break
            
        ## Update x^(k) after check in case of break
        x_coo[v] += r 
        
    output = {'x': x_coo,           # final iterate x
              'D': D_rec[:(q+1)],   # evolution of the cost D
              'R': R_rec[:(q+1)],   # evolution of the cost R
              'q': q }              # index of the final iteration
    
    return output




## Coordinate descent on quadratic map D with rescaling of the iterates (simple rescaling).
# Inputs: number of iterations, matrix Q, vector c, constant term D0.
# Outputs: candidate solution x, record of D map, number of iterations taken.  
def D_GSL_Solver_Rescale(n_iter,   # maximum number of iterations
                         Q_mat,    # matrix Q
                         c_vec,    # vector c 
                         D0):      # value of D(0)
    
    N = len(c_vec)
    sqrt_diag_Q_inv = g_inv_vec(np.sqrt(np.abs(np.diag(Q_mat))))
    
    D_rec = np.zeros(n_iter+1, dtype=float) # record D(v)
    x_coo = np.zeros(N, dtype=float)        
    Qx_vec = np.zeros(N, dtype=float)
    xQx = 0
    cx = 0
    res = - c_vec
    D_rec[0] = D0

    
    for q in range(0,n_iter):
    
        ## Selection 
        v = np.argmax(np.abs(res * sqrt_diag_Q_inv))
        r = -res[v] / Q_mat[v,v]
        
        if res[v]==0 : 
            #print('zero grad')
            q = max(0,q-1) ; break # minimum is reached
              
        
        ## Update 
        xQx += (r**2) * Q_mat[v,v] + 2 * r * Qx_vec[v] 
        Qx_vec += r * Q_mat[:,v]  
        cx += r * c_vec[v] 
            
        ## Update x^(k) after check in case of break
        x_coo[v] += r 
        
        ## Rescale
        sx = cx / xQx
        x_coo = sx * x_coo
        Qx_vec = sx * Qx_vec
        xQx = (sx**2) * xQx
        cx = sx * cx
        res = Qx_vec - c_vec 
        
        D_rec[q+1] = D0 + xQx - 2 * cx
        if D_rec[q+1] > D_rec[q] : 
            #print('cost increase')
            q = max(0,q-1) ; break
        if D_rec[q+1] < 0 : 
            #print('negative cost')
            q = max(0,q-1) ; break
        
  
        
    output = {'x': x_coo,           # final iterate x
              'D': D_rec[:(q+1)],   # evolution of the cost D
              'q': q }              # index of the final iteration
    
    return output




## Conjugate gradient for the minimisation of the quadratic map D.
# Inputs: number of iterations, matrix Q, vector c, constant term D0.
# Outputs: candidate solution x, record of D map, number of iterations taken.  
def CG_Solver(n_iter,   # maximum number of iterations
              Q_mat,    # matrix Q
              c_vec,    # vector c 
              D0):      # value of D(0) 
    
    N = len(c_vec)
    D_rec = np.zeros(n_iter+1, dtype=float) # record D(v)
    
    ## Initialisation    
    x_coo = np.zeros(N, dtype=float)
    grad = -c_vec
    D_rec[0] = D0
    dir_vec = -grad
    
    ## Iterate
    for q in range(0,n_iter):
        
        if np.max(np.abs(grad)) == 0: 
            #print('zero grad')
            q = max(0,q-1) ; break # solution
        
        ## Optimal step size
        Q_dir = Q_mat @ dir_vec
        denom = np.sum(dir_vec * Q_dir)
        oss = -np.sum(grad * dir_vec)/denom 
        
        ## Update
        x_coo += oss * dir_vec
        grad += oss * Q_dir
        
        D_rec[q+1] = D0 + np.sum(x_coo*(Q_mat@x_coo)) - 2 * np.sum(c_vec*x_coo)
        
        if D_rec[q+1] >= D_rec[q] : 
            #print('cost increase')
            q = max(0,q-1) ; break
        if D_rec[q+1] <= 0 : 
            #print('negative cost')
            q = max(0,q-1) ; break
            
        dir_step = np.sum(grad*Q_dir)/denom 
        dir_vec = - grad + dir_step * dir_vec
        
    output = {'x': x_coo,               # final iterate x
              'D_rec': D_rec[:(q+1)],   # evolution of the cost D
              'q': q }                  # index of the final iteration
    
    return output   


