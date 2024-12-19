import numpy as np
import utils
import matplotlib.pyplot as plt
import imageio

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
    
    def beta(x, mu_, sig_, alpha_):
        L = np.linalg.cholesky(np.linalg.inv(sig_))
        sign, log_det = np.linalg.slogdet(L)
        
        diff = x - mu_[None, :]
        norm = diff @ L
        sq_norm = np.sum(norm**2,axis=1)

        
        beta = -0.5 * ( sq_norm + m * np.log(2*np.pi) ) + sign * log_det + np.log(alpha_)
        
        return beta

    for it in range(max_iter):
        # TODO: Implement (9) - (11)

        llh = np.zeros((K,N)) 
        for k in range(K):
            llh[k] = beta(X, mus[k], sigmas[k], alphas[k])
    
        #ok - LogSumExp
        max_llh = np.max(llh, axis=0)
        diff_llh = llh - max_llh
        exp_diff_llh = np.exp(diff_llh)
        sum_exp_diff_llh = np.sum(exp_diff_llh, axis=0)
        log_sum_exp_diff_llh = max_llh + np.log(sum_exp_diff_llh)
        
        #gama
        log_wk = llh-log_sum_exp_diff_llh
        wk = np.exp(log_wk)

        # Compute Nk - sum of gamas
        Nk = np.sum(wk, axis=1)
        
        # recompute params
        alphas = Nk / N

        mus = ( wk[...,None] * X[None,...] ).sum(1) / Nk[:,None]                                # K x M*M
        
        diff = (X[None,...] - mus[:,None,:])                                                   # K x N x M*M
        sigmas = np.einsum( 'KNi, KNj -> Kij', wk[:,:,None] * diff, diff) / Nk[:,None,None]   # K x M*M x M*M 
        sigmas = sigmas + epsilon * np.eye(m) 

        if it % show_each == 0 and plot:
            print(it)

        if it  == max_iter - 1 and plot:
            utils.plot_gmm(X, alphas, mus, sigmas)



    #
    # Caluclates the probability under the fitted models. 
    #
    # print("Our alpha: ", alphas)
    # our = np.zeros((K,N)) 
    # for k in range(K):
    #     our[k] = beta(X, mus[k], sigmas[k], alphas[k])

    # max_our = np.max(our, axis=0)
    # diff_our = our - max_our
    # exp_diff_our = np.exp(diff_our)
    # sum_exp_diff_our = np.sum(exp_diff_our, axis=0)
    # log_sum_exp_diff_our = max_our + np.log(sum_exp_diff_our)

    # prof_alphas = np.array([0.2, 0.8]) # taken from utils
    # print("Prof alpha: ", prof_alphas) 
    # prof = np.zeros((K,N)) 
    # for k in range(K):
    #     prof[k] = beta(X, utils._mus[k], utils._sigmas[k], prof_alphas[k])

    # max_prof = np.max(prof, axis=0)
    # diff_prof = prof - max_prof
    # exp_diff_prof = np.exp(diff_prof)
    # sum_exp_diff_prof = np.sum(exp_diff_prof, axis=0)
    # log_sum_exp_diff_prof = max_prof + np.log(sum_exp_diff_prof)

    # print(np.exp(log_sum_exp_diff_our) - np.exp(log_sum_exp_diff_prof))

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

    def beta(x, mu_, sig_, alpha_):
        L = np.linalg.cholesky(np.linalg.inv(sig_))
        sign, log_det = np.linalg.slogdet(L)
        
        diff = x - mu_[None, :]
        norm = diff @ L
        sq_norm = np.sum(norm**2,axis=1)

        
        beta = -0.5 * ( sq_norm + m * np.log(2*np.pi) ) + sign * log_det + np.log(alpha_)
        
        return beta
    
    # TODO: Precompute A, b (26)
    
    A = np.zeros((K,m,m))
    b = np.zeros((K,m))

    A = np.linalg.inv(lamda*np.eye(m) + E.T[None,:] @ (np.linalg.inv(sigmas)) @ E[None,:])
    b = np.linalg.inv(sigmas) @ E[None,:] @ mus[:,:,None]
    b = b.squeeze(-1)
    
    tmp = lamda*y
    N = x_est.shape[0]
    
    for it in range(max_iter):
        # TODO: Implement Line 3, Line 4 of Algorithm 1
        
        betas=np.zeros((K, N))
        for k in range(K):
            betas[k] = beta(x_est @ E , mus[k], sigmas[k], alphas[k])
        
        k_max = np.argmax(betas, axis =0)
        x_tilde = A[k_max]

        x_tilde =(x_tilde @ ( tmp + b[k_max])[:,:,None]).squeeze(-1)

        x_est = alpha * x_est + (1 - alpha) * x_tilde

        if not test and it == max_iter-1:
            u = utils.patches_to_image(x_est, x.shape, w)
            print(f"it: {it+1:03d}, psnr(u, y)={utils.psnr(u, x):.2f}")

    return utils.patches_to_image(x_est, x.shape, w)


def benchmark(K: int = 10, w: int = 5):
    for i in range(1, 5):
        utils.imsave(f'./test/img{i}_out.png', denoise(i, K, w, test=True))


def train(use_toy_data: bool = True, K: int = 2, w: int = 5):
    data = np.load('./toy.npy') if use_toy_data else utils.train_data(w)
    # Plot only if we use toy data
    alphas, mus, sigmas = expectation_maximization(data, K=K, plot=use_toy_data)
    # Save only if we dont use toy data
    if not use_toy_data:
        utils.save_gmm(K, w, alphas, mus, sigmas)

def models_test():
    k = [2,5,10,10,10,15]
    w = [5,5,3,5,7,5]
    num = len(k)
    fig, axes = plt.subplots(2, 3, figsize=(16,12)) 
    axes = axes.flatten()
    for i in range(num):
        denoised = denoise(1, k[i], w[i], test=False,sigma=0.1)
        denoised = (np.clip(denoised, 0, 1) * 255.).astype(np.uint8)
        axes[i].imshow(denoised,cmap='gray')
        axes[i].axis('off')
        param_text = f"K: {k[i]}, w: {w[i]}\n"
        axes[i].text(0.5, -0.15, param_text, ha='center', va='top', transform=axes[i].transAxes, fontsize=10)
        imageio.imsave(f'./model_test/img{i}.png', denoised)
    fig.savefig('model_res.png')

if __name__ == "__main__":
    do_training = False
    # Use the toy data to debug your EM implementation
    use_toy_data = False
    # Parameters for the GMM: Components and window size, m = w ** 2
    # Use K = 2 for toy/debug model
    K = 15
    w = 5
    #models_test()
    benchmark(K, w)
    # if do_training:
    #     train(use_toy_data, K, w)
    # else:
    #     # fig, axes = plt.subplots(5, 3, figsize=(10, 15))  
    #     # axes[0, 0].set_title('Noisy')
    #     # axes[0, 1].set_title('Denoised')
    #     # axes[0, 2].set_title('Ground Truth')
    #     for i in range(1, 6):
    #         denoised = denoise(i, K, w, test=False,sigma=0.1)
    #         x, y = utils.validation_data(i, sigma=0.1, seed=1, w=w)
    #         noisy = utils.patches_to_image(y, x.shape, w)
    #         # axes[i-1,0].imshow(noisy,cmap='gray')
    #         # axes[i-1,1].imshow(denoised,cmap='gray')
    #         # axes[i-1,2].imshow(x,cmap='gray')

    #         utils.imsave(f'./validation/img{i}_out.png', denoised)
    #     # for ax_row in axes:
    #     #     for ax in ax_row:
    #     #         ax.axis('off')
    #     # plt.tight_layout()
    #     # plt.savefig("validation_0.1.png")

    # # If you want to participate in the challenge, you can benchmark your model
    # # Remember to upload the images in the submission.
    # benchmark(K, w)
