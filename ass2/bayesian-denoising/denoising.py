import numpy as np
import utils


def expectation_maximization(
    X: np.ndarray,
    K: int,
    max_iter: int = 50,
    plot: bool = False,
    show_each: int = 5,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Number of data points, features
    N, m = X.shape
    # Init: Uniform weights, first K points as means, identity covariances
    alphas = np.full((K,), 1. / K)
    mus = X[:K]
    sigmas = np.tile(np.eye(m)[None], (K, 1, 1))

    
    def pdf(x,mu_,sig_,alpha_):
        d = x.shape[-1]
        sigma_offset = sig_ #+ 1e-6 * np.eye(d)
        L = np.linalg.inv(np.linalg.inv(sigma_offset)) ####??????????
        sigma_det = 2 * np.sum(np.log(np.diag(L)))  
        fact = -0.5 * (d * np.log(2 * np.pi) + sigma_det)#log sigma det
        diff = x - mu_[None, :]
        inv_sig = np.linalg.inv(sigma_offset)
        tmp = np.einsum('ij,Nj->Ni',inv_sig,diff)
        tmp = np.einsum('Ni,Ni->N',diff,tmp)
        log_prob = fact + (-0.5) * tmp + np.log(alpha_)
        return log_prob
    
    def beta(x,mu_,sig_,alpha_):
        m = x.shape[1]
        L = np.linalg.inv(np.linalg.cholesky(sig_))
        #L = np.linalg.cholesky(np.linalg.inv(sig_))
        sign, log_det = np.linalg.slogdet(L)
        diff = x - mu_[None, :]
        #norm = (L @ diff.T).T
        norm = diff @ L.T
        sq_norm = np.sum(norm**2,axis=1)
        beta = -0.5*(sq_norm+m*np.log(2*np.pi)) + sign*log_det + np.log(alpha_)
        
        return beta

    for it in range(max_iter):
        print(it)
        # TODO: Implement (9) - (11)
        alphas = alphas
        mus = mus
        sigmas = sigmas
    
        llh = np.zeros((K,N)) 
        for k in range(K):
            #llh[k] = pdf(X,mus[k],sigmas[k],alphas[k])
            llh[k] = beta(X,mus[k],sigmas[k],alphas[k])
    
        #ok - LogSumExp
        max_llh = np.max(llh, axis=0)
        diff_llh = llh - max_llh
        exp_diff_llh = np.exp(diff_llh)
        sum_exp_diff_llh = np.sum(exp_diff_llh, axis=0)
        log_sum_exp_diff_llh = max_llh + np.log(sum_exp_diff_llh)
        
        #llh_all.append(-np.sum(log_sum_exp_diff_llh))#?

        #gama
        log_wk = llh-log_sum_exp_diff_llh
        wk = np.exp(log_wk)
        # Compute Nk - sum of gamas
        Nk = np.sum(wk, axis=1)
        # recompute params
        mus = (wk[...,None]*X[None,...]).sum(1)/Nk[:,None] #KxM*M
        diff = (X[None,...] - mus[:,None,:]) #KxNxM*M
        sigmas = np.einsum('KNi,KNj->Kij',wk[:,:,None]*diff,diff)/Nk[:,None,None] #KxM*MxM*M
        alphas = Nk/N 
        sigmas = sigmas + epsilon*np.eye(m)
        # if it >= 2 and np.abs(llh_all[-2] - llh_all[-1]) <= epsilon:
        #     print('EM algo. terminated in iter %i' %it)
        if it == max_iter - 1 and plot:
            utils.plot_gmm(X, alphas, mus, sigmas)

        # if it % show_each == 0 and plot:
        #     utils.plot_gmm(X, alphas, mus, sigmas)

    return alphas, mus, sigmas


def denoise(
    index: int = 1,
    K: int = 10,
    w: int = 5,
    alpha: float = 0.5,
    max_iter: int = 30,
    test: bool = False,
    sigma: float = 0.1
):
    alphas, mus, sigmas = utils.load_gmm(K, w)
    precs = np.linalg.inv(sigmas)
    precs_chol = np.linalg.cholesky(precs)  # "L" in the assignment sheet
    if test:
        # This is really two times `y` since we dont have access to `x` here
        x, y = utils.test_data(index)
    else:
        x, y = utils.validation_data(index, sigma=sigma, seed=1, w=w)
    # x is image-shaped, y is patch-shaped
    # Initialize the estimate with the noisy patches
    x_est = y.copy()
    m = w ** 2
    lamda = 1 / sigma ** 2
    E = np.eye(m) - np.full((m, m), 1 / m)

    def beta(x,mu_,sig_,alpha_):
        m = x.shape[1]
        L = np.linalg.inv(np.linalg.cholesky(sig_))
        #L = np.linalg.cholesky(np.linalg.inv(sig_))
        sign, log_det = np.linalg.slogdet(L)
        diff = x - mu_[None, :]
        #norm = (L @ diff.T).T
        norm = diff @ L.T
        sq_norm = np.sum(norm**2,axis=1)
        beta = -0.5*(sq_norm+m*np.log(2*np.pi)) + sign*log_det + np.log(alpha_)
        
        return beta
    # TODO: Precompute A, b (26)
    
    A = np.zeros((K,m,m))
    b = np.zeros((K,m))

    A = np.linalg.inv(lamda*np.eye(m) + E.T[None,:]@(np.linalg.inv(sigmas))@E[None,:])
    b = np.linalg.inv(sigmas)@E[None,:] @ mus[:,:,None]
    b = b.squeeze(-1)
    
    tmp = lamda*y
    N = x_est.shape[0]
    
    for it in range(max_iter):
        # TODO: Implement Line 3, Line 4 of Algorithm 1
        
        betas=np.zeros((K))
        for k in range(K):
            betas[k] = np.sum(beta(x_est@E,mus[k],sigmas[k],alphas[k]),axis=0)
        
        k_max = np.argmax(betas)
        x_tilde = A[k_max][None,:]

        x_tilde =(x_tilde@(tmp+b[k_max][None,:])[:,:,None]).squeeze(-1)

        x_est = alpha * x_est + (1 - alpha) * x_tilde

        if not test:
            u = utils.patches_to_image(x_est, x.shape, w)
            print(f"it: {it+1:03d}, psnr(u, y)={utils.psnr(u, x):.2f}")

    return utils.patches_to_image(x_est, x.shape, w)


def benchmark(K: int = 10, w: int = 5):
    for i in range(1, 5):
        utils.imsave(f'./test/img{i}_out_my.png', denoise(i, K, w, test=True))


def train(use_toy_data: bool = True, K: int = 2, w: int = 5):
    data = np.load('./toy.npy') if use_toy_data else utils.train_data(w)
    # Plot only if we use toy data
    alphas, mus, sigmas = expectation_maximization(data, K=K, plot=use_toy_data)
    # Save only if we dont use toy data
    if not use_toy_data:
        utils.save_gmm(K, w, alphas, mus, sigmas)


if __name__ == "__main__":
    do_training = False
    # Use the toy data to debug your EM implementation
    use_toy_data = True
    # Parameters for the GMM: Components and window size, m = w ** 2
    # Use K = 2 for toy/debug model
    K = 2
    w = 5
    if do_training:
        train(use_toy_data, K, w)
    else:
        for i in range(1, 6):
            denoise(i, K, w, test=False)

    # If you want to participate in the challenge, you can benchmark your model
    # Remember to upload the images in the submission.
    benchmark(K, w)
