import numpy as np
import h5py
import gpflow as gp
from EI_func import EI_bo
from PI_func import PI_bo
from GP_UCB_func import GP_UCB_bo
from Random_func import Random_bo
from TStdR import TStdR_bo
from TStdR_with_discard import TStdR_with_discard_bo
from EI_weighted_TStdR_with_discard import EI_weighted_TStdR_with_discard_bo
from EI_factored_TStdR_with_discard import EI_factored_TStdR_with_discard_bo
from PI_factored_TStdR_with_discard import PI_factored_TStdR_with_discard_bo
from plots import *

with h5py.File('./datasets/1d/gp_sample_rbf_l_0.5_v_1.0.h5', 'r') as hf:
    X= np.array(hf.get('X')); Y= np.array(hf.get('y_sample'))

iter=1
random_indexes=np.random.permutation(np.arange(X.shape[0]))[0:iter]

loss_ei= 0; loss_pi= 0; loss_ucb= 0; loss_rand= 0; loss_tsdtr=0; loss_tsdtr_with_discard= 0; loss_ei_weighted_tsdtr_with_discard= 0
loss_ei_factored_tsdtr_with_discard= 0; loss_pi_factored_tsdtr_with_discard= 0


for i in range(iter):

    ls= 0.5; var= 1.0; budget= 10; kernel= gp.kernels.RBF(lengthscales= ls, variance= var)

    # index0= np.random.choice(np.arange(X.shape[0]))
    index0= random_indexes[i]

    loss_list_ei, Xt_ei, Yt_ei, model_ei= EI_bo(X, Y, kernel, budget, index0, noise= 10**(-4))
    loss_ei+= np.array(loss_list_ei)/iter

    loss_list_pi, Xt_pi, Yt_pi, model_pi= PI_bo(X, Y, kernel, budget, index0, noise= 10**(-4))
    loss_pi += np.array(loss_list_pi)/ iter

    kapa= 3.0
    loss_list_ucb, Xt_ucb, Yt_ucb, model_ucb= GP_UCB_bo(X, Y, kernel, budget, index0, kapa, noise= 10**(-4))
    loss_ucb+= np.array(loss_list_ucb) / iter

    loss_list_rand, Xt_rand, Yt_rand, model_rand= Random_bo(X, Y, kernel, budget, index0, noise= 10**(-4))
    loss_rand += np.array(loss_list_rand) / iter

    loss_list_tstdr, Xt_tsdtr, Yt_tsdtr, model_tsdtr= TStdR_bo(X, Y, kernel, budget, index0, noise=10 ** (-4), plot=False)
    loss_tsdtr += np.array(loss_list_tstdr)/iter

    loss_list_tsdtr_discard, Xt_tsdtr_discard, Yt_tsdtr_discard, model_tsdtr_discard= TStdR_with_discard_bo(X, Y, kernel, budget, index0, noise=10 ** (-4), plot=False)
    loss_tsdtr_with_discard += np.array(loss_list_tsdtr_discard)/iter

    # loss_list_EI_weighted_TStdR_discard, Xt_tsdtr_EI_weighted_discard, Yt_tsdtr_EI_weighted_discard,model_tsdtr_EI_weighted_discard \
    #             = EI_weighted_TStdR_with_discard_bo(X, Y, kernel, budget, index0, kapa=1, noise=10 ** (-4), plot=False)
    # loss_ei_weighted_tsdtr_with_discard+= np.array(loss_list_EI_weighted_TStdR_discard)/iter

    loss_list_EI_factored_TStdR_discard, Xt_tsdtr_EI_factored_discard, Yt_tsdtr_EI_factored_discard,model_tsdtr_EI_factored_discard \
                = EI_factored_TStdR_with_discard_bo(X, Y, kernel, budget, index0, kapa=1, noise=10 ** (-4), plot=False)
    loss_ei_factored_tsdtr_with_discard+= np.array(loss_list_EI_factored_TStdR_discard)/iter

    loss_list_PI_factored_TStdR_discard, Xt_tsdtr_PI_factored_discard, Yt_tsdtr_PI_factored_discard,model_tsdtr_PI_factored_discard \
                = PI_factored_TStdR_with_discard_bo(X, Y, kernel, budget, index0, kapa=1, noise=10 ** (-4), plot=False)
    loss_pi_factored_tsdtr_with_discard+= np.array(loss_list_PI_factored_TStdR_discard)/iter

# print('loss_ei:{}\n loss_pi:{}\n loss_ucb:{}\n loss_rand:{}\n loss_tsdtr:{}\n loss_tsdtr_with_discard:{}\n loss_ei_weighted_tsdtr_with_discard:{}\n'
#       'loss_ei_factored_tsdtr_with_discard:{}\n loss_pi_factored_tsdtr_with_discard:{}'.\
#       format(loss_ei, loss_pi, loss_ucb, loss_rand, loss_tsdtr, loss_tsdtr_with_discard, loss_ei_weighted_tsdtr_with_discard, \
#              loss_ei_factored_tsdtr_with_discard, loss_pi_factored_tsdtr_with_discard))


dict= {'ei':loss_ei, 'pi': loss_pi, 'ucb':loss_ucb, 'rand':loss_rand, 'tsdtr':loss_tsdtr, 'tsdtr_discard':loss_tsdtr_with_discard,\
       'ei_weighted_tsdtr_with_discard': loss_ei_weighted_tsdtr_with_discard, 'ei_factored_tsdtr_with_discard':loss_ei_factored_tsdtr_with_discard, \
       'pi_factored_tsdtr_with_discard': loss_pi_factored_tsdtr_with_discard}

comparison_plot(budget, loss_ei, loss_pi, loss_ucb, loss_rand, loss_tsdtr, loss_tsdtr_with_discard,loss_ei_factored_tsdtr_with_discard,\
                loss_pi_factored_tsdtr_with_discard)



