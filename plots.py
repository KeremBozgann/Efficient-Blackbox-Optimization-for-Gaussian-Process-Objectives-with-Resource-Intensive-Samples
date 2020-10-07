import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('../bo_cost_budget_cont_domain')
from util import create_grid

def plot_posterior(X, y_true, model):

    u_pred, sigma_pred= model.predict_f(X)

    X1, X2= X[:,0].reshape(-1,1), X[:,1].reshape(-1,1)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(X1,X2, u_pred, color= 'red', label= 'posterior mean')
    ax.scatter(X1,X2, y_true, color= 'green', label= 'true value')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y');

def plot_3D(X1,X2,y, title):

    fig = plt.figure()

    ax = Axes3D(fig)
    ax.set_title(title)
    ax.scatter3D(X1, X2, y)

def plot_2D(X,y):

    plt.figure()
    plt.plot(X, y)

def plot_1D(Xt, Yt, xt, yt, X, u ,var):

    plt.figure()
    plt.scatter(Xt, Yt, color='orange', marker='X')
    plt.scatter(xt, yt, color='blue', marker='X')
    plt.plot(X, u, color='red')
    plt.fill_between(X.flatten(), (u + np.sqrt(var)).flatten(), (u - np.sqrt(var)).flatten(), color='lightgrey',
                     alpha='0.5')
    plt.xlabel('x');
    plt.ylabel('y');
    plt.show()

def plot_evaluated_points(Xt, Yt, X, Y):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(Xt[:,0], Xt[:,1], Yt[:,0], color= 'black', marker='X')
    ax.scatter3D(X[:,0], X[:,1], Y[:,0], color= 'orange', alpha= 0.5)

def plot_loss_vs_evaluation(loss_list, evaluation_budget):

    plt.figure()
    plt.plot(np.arange(evaluation_budget), np.array(loss_list), color='black')
    plt.scatter(np.arange(evaluation_budget), np.array(loss_list), color='black')
    plt.xlabel('evaluation');
    plt.ylabel('loss')
    plt.show()

def plot_loss_vs_time(loss_list, cum_cost_list):

    plt.figure()
    plt.plot(np.squeeze(cum_cost_list), np.array(loss_list), color= 'black')
    plt.scatter(np.squeeze(cum_cost_list), np.array(loss_list), color='black')
    plt.xlabel('cum_cost_list'); plt.ylabel('loss')
    plt.show()

def posterior_acquisition_chosen_point_evaluated_points(X, Y, Xt, Yt, x_chosen, y_chosen, mu, sigma,Acq):

    plt.figure()
    plt.scatter(X, np.zeros(X.shape[0], ), color='black', marker='X')
    plt.scatter(Xt, Yt, color='orange', marker='X')
    plt.scatter(x_chosen, y_chosen, color='blue', marker='X')
    plt.plot(X, mu, color='red')
    plt.fill_between(X[:, 0], mu[:, 0] + sigma[:, 0], mu[:, 0] - sigma[:, 0], color='grey', alpha=0.5)
    plt.plot(X,Y, color='green')
    plt.plot(X, Acq, color='black' )
    plt.xlabel('x');
    plt.ylabel('y')

    plt.show()

def cost_posterior_chosen_point_evaluated_points(X, Y_cost, Xt, Yt_cost, xt_cost, yt_cost, u_cost, sigma_cost ):

    plt.figure()
    plt.scatter(X, np.zeros(X.shape[0], ), color='black', marker='X')
    plt.scatter(Xt, Yt_cost, color='orange', marker='X')
    plt.scatter(xt_cost, yt_cost, color='blue', marker='X')
    plt.plot(X, u_cost, color='red')
    plt.fill_between(X[:, 0], u_cost[:, 0] + sigma_cost[:, 0], u_cost[:, 0] - sigma_cost[:, 0], color='grey', alpha=0.5)
    plt.plot(X,Y_cost, color='green')
    plt.xlabel('x')
    plt.ylabel('y_cost')

    plt.show()

def comparison_plot(budget, loss_ei, loss_pi, loss_ucb, loss_rand, loss_tsdtr, loss_tsdtr_with_discard, \
                    loss_ei_factored_tsdtr_with_discard, loss_pi_factored_tsdtr_with_discard):

    plt.figure()

    plt.plot(np.arange(budget), loss_ei, label='ei')
    plt.scatter(np.arange(budget), loss_ei)

    plt.plot(np.arange(budget), loss_pi, label='pi')
    plt.scatter(np.arange(budget), loss_pi)

    plt.plot(np.arange(budget), loss_ucb, label='ucb')
    plt.scatter(np.arange(budget), loss_ucb)

    plt.plot(np.arange(budget), loss_rand, label='rand')
    plt.scatter(np.arange(budget), loss_rand)

    plt.plot(np.arange(budget), loss_tsdtr, label='tsdtr')
    plt.scatter(np.arange(budget), loss_tsdtr)

    plt.plot(np.arange(budget), loss_tsdtr_with_discard, label='tsdtr_discard')
    plt.scatter(np.arange(budget), loss_tsdtr_with_discard)

    # plt.plot(np.arange(budget), loss_ei_weighted_tsdtr_with_discard, label='tsdtr_ei_weight_discard')
    # plt.scatter(np.arange(budget), loss_ei_weighted_tsdtr_with_discard)

    plt.plot(np.arange(budget), loss_ei_factored_tsdtr_with_discard, label='tsdtr_ei_fact_discard')
    plt.scatter(np.arange(budget), loss_ei_factored_tsdtr_with_discard)

    plt.plot(np.arange(budget), loss_pi_factored_tsdtr_with_discard, label='tsdtr_pi_fact_discard')
    plt.scatter(np.arange(budget), loss_pi_factored_tsdtr_with_discard)

    plt.xlabel('iteration')
    plt.ylabel('average loss')
    plt.legend()
    plt.show()

def comparison_plot_with_dict(dict):

    plt.figure()
    plt.title('loss vs iteration')
    for method in dict:

        x_list = np.arange(len(dict[method]))
        plt.plot(x_list, dict[method], label= method)
        plt.scatter(x_list, dict[method], label= method)

    plt.xlabel('iteration')
    plt.ylabel('average loss')
    plt.legend()
    plt.show()

def comparison_cost(cost_dict):

    plt.figure()
    plt.title('loss vs iteration')
    for method in cost_dict:

        x_list = np.arange(len(cost_dict[method]))
        plt.plot(x_list, cost_dict[method], label= method)
        plt.scatter(x_list, cost_dict[method], label= method)

    plt.xlabel('iteration')
    plt.ylabel('average cum cost')
    plt.legend()
    plt.show()

def plot_cost_and_target(X, Y, Y_cost):

    fig, axs= plt.subplots(2)

    axs[0].plot(X, Y, color= 'blue', label= 'target function')
    axs[0].set(xlabel='x', ylabel='target'); axs[0].legend(); #axs[0].set_title('target')

    axs[1].plot(X, Y_cost, color= 'green', label= 'cost function')
    axs[1].set(xlabel= 'x', ylabel= 'cost'); axs[1].legend(); #axs[1].set_title('cost')

def plot_posterior_and_true_target(X, u_X, Y, Xt, Yt, index_max):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('target')
    ax.scatter3D(X[:,0], X[:,1],Y[:,0], color= 'green', label= 'true target')
    ax.scatter3D(X[:,0], X[:,1],u_X[:,0], color= 'blue', label= 'posterior target')
    ax.scatter3D(Xt[:,0], Xt[:,1],Yt[:,0],color= 'orange', label= 'evaluated points')
    ax.scatter3D(X[index_max, :][:,0], X[index_max, :][:,1], Y[index_max, :][:,0], color= 'blue', label= 'point chosen for eval')

def plot_cost_posterior_and_true_cost(X, u_cost, Y_cost, Xt, Yt_cost, index_max):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('cost')
    ax.scatter3D(X[:,0], X[:,1],Y_cost[:,0], color= 'green', label= 'true cost')
    ax.scatter3D(X[:,0], X[:,1],u_cost[:,0], color= 'blue', label= 'posterior cost')
    ax.scatter3D(Xt[:,0], Xt[:,1],Yt_cost[:,0],color= 'orange', label= 'evaluated points')
    ax.scatter3D(X[index_max, :][:,0], X[index_max, :][:,1], Y_cost[index_max, :][:,0], color= 'blue', label= 'point chosen for eval')


def plot_posterior_and_true_target_and_acqusitions(X, Y, Y_cost, Xt, Yt, Yt_cost, xt,
                           u_X, sigma_X, u_latent_cost, sigma_latent_cost, Acq_dict,index_max, plot_cost):




    Acq_normalized_dict={}

    for method in Acq_dict:
        Acq_normalized_dict[method]= Acq_dict[method]/np.sum(Acq_dict[method],axis=0)

    # Acq_normalized= Acq_normalized_dict[list(Acq_normalized_dict.keys())[0]]

    if plot_cost== True:
        fig, axs = plt.subplots(3)
    else:
        fig, axs = plt.subplots(2)

    '''target posterior'''
    axs[0].set_title('target')
    axs[0].scatter(Xt, Yt, color='orange', marker='X') #evaluated points
    axs[0].scatter(xt, u_X[index_max,:], color='blue', marker='X') #chosen point
    axs[0].plot(X, u_X, color='red', label= 'target posterior') #posterior
    axs[0].fill_between(X[:, 0], u_X[:, 0] + sigma_X[:, 0], u_X[:, 0] - sigma_X[:, 0], color='grey', alpha=0.5)
    axs[0].plot(X,Y, color='green', label= 'true target')  #target function
    axs[0].set_xlabel('x'); axs[0].set_ylabel('y')
    axs[0].legend()


    '''acqusitions'''
    axs[1].set_title('acqusitions')

    for method in Acq_normalized_dict:

        axs[1].plot(X, Acq_normalized_dict[method], label= method)
        # axs[1].plot(X, Acq_normalized, color='green', label= 'EI_cool_normalized')
        # axs[1].plot(X, Acq_EI_pu_normalized, color= 'blue', label= 'EI_pu_normalized')
        # axs[1].plot(X, Acq_EI_normalized, color='orange', label='EI_normalized')
    axs[1].scatter(X, np.zeros(X.shape[0], ), color='black', marker='X')
    axs[1].set_xlabel('x'); axs[1].set_ylabel('acq')
    axs[1].legend()

    if plot_cost==True:

        '''cost posterior'''
        u_cost= np.exp(u_latent_cost)

        axs[2].set_title('cost')
        axs[2].scatter(Xt, Yt_cost, color= 'orange', marker= 'X') #evaluated cost points
        axs[2].scatter(xt, u_cost[index_max, :], color= 'blue', marker= 'X')
        axs[2].plot(X, u_cost, color= 'red', label= 'cost posterior') #posterior
        axs[2].fill_between(X[:,0], np.exp(u_latent_cost[:,0]+sigma_latent_cost[:,0]),
                            np.exp(u_latent_cost[:,0]-sigma_latent_cost[:,0]), color = 'grey', alpha= 0.5)
        axs[2].plot(X, Y_cost, color= 'green', label= 'true cost') #target function
        axs[2].legend()
        axs[2].set_xlabel('x'); axs[2].set_ylabel('cost')
    
    plt.show()

def plot_average_loss(loss_dict):

    plt.figure()
    for method in loss_dict:
        loss = loss_dict[method][0]; count= loss_dict[method][1]
        cost_grid= loss_dict[method][2]

        avg_loss= loss/count
        plt.plot(cost_grid,avg_loss, label= method)
        plt.scatter(cost_grid, avg_loss)

    plt.legend()
    plt.show()


def plot_and_save_average_loss_and_std(loss_dict, folder, exp_name):

    plt.figure()
    for method in loss_dict:
        '''get mean'''
        loss = loss_dict[method][0]; count= loss_dict[method][1]
        cost_grid= loss_dict[method][2]
        avg_loss= loss/count

        '''get std'''
        std = loss_dict[method][6]

        # plt.plot(cost_grid,avg_loss, label= method, alpha= 0.6)
        plt.scatter(cost_grid, np.log10(avg_loss), alpha= 0.6)
        plt.errorbar(cost_grid, np.log10(avg_loss), yerr= std, linestyle='-', elinewidth=1
                   ,linewidth=2, label=method,  alpha= 0.6)

    plt.xlabel('cost')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('../Results/' + folder + '/' + exp_name)
    plt.show()

def plot_and_save_average_loss_y_best(loss_dict, folder, exp_name):
    plt.figure()
    for method in loss_dict:
        loss = loss_dict[method][0]; count= loss_dict[method][1]
        cost_grid= loss_dict[method][2]
        f_best= loss_dict[method][3]
        avg_loss= f_best/count
        plt.plot(cost_grid,avg_loss, label= method, alpha= 0.6)
        plt.scatter(cost_grid, avg_loss, alpha= 0.6)
    plt.xlabel('cost')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('../Results/' + folder + '/' + exp_name)
    plt.show()

def plot_posterior_and_acquisitions_for_continuous_domain(X, Xt, Yt, Yt_cost, xt,
               u_X, sigma_X, u_latent_cost, sigma_latent_cost, Acq_dict, plot_cost, model, latent_cost_model, Y_cost, Y,
                                                          plot_true_cost= False, plot_true_targ= False):

    ut, vart= model.predict_f(xt)
    ut_latent_cost, vart_latent_cost= latent_cost_model.predict_f(xt); ut_latent_cost= ut_latent_cost.numpy()
    ut_cost= np.exp(ut_latent_cost)

    Acq_normalized_dict = {}

    for method in Acq_dict:
        if any(Acq_dict[method]<0):
            Acq_dict[method]+= -np.min(Acq_dict[method])

        # Acq_normalized_dict[method] = Acq_dict[method] / np.sum(Acq_dict[method], axis=0)
        Acq_normalized_dict[method]= Acq_dict[method]
    # Acq_normalized= Acq_normalized_dict[list(Acq_normalized_dict.keys())[0]]

    if plot_cost == True:
        fig, axs = plt.subplots(3)
    else:
        fig, axs = plt.subplots(2)

    '''target posterior'''
    axs[0].set_title('target')
    axs[0].scatter(Xt, Yt, color='orange', marker='X')  # evaluated points
    axs[0].scatter(xt, ut.numpy(), color='blue', marker='X')  # chosen point
    axs[0].plot(X, u_X, color='red', label='target posterior')  # posterior
    axs[0].fill_between(X[:, 0], u_X[:, 0] + sigma_X[:, 0], u_X[:, 0] - sigma_X[:, 0], color='grey', alpha=0.5)
    if plot_true_targ:
        '''true cost'''
        axs[0].set_title('true target')
        axs[0].plot(X, Y, color='green', label='true cost')  # true cost function

    axs[0].set_xlabel('x');
    axs[0].set_ylabel('y')
    axs[0].legend()

    '''acqusitions'''
    axs[1].set_title('acqusitions')

    for method in Acq_normalized_dict:
        axs[1].plot(X, Acq_normalized_dict[method], label=method)
        # axs[1].plot(X, Acq_normalized, color='green', label= 'EI_cool_normalized')
        # axs[1].plot(X, Acq_EI_pu_normalized, color= 'blue', label= 'EI_pu_normalized')
        # axs[1].plot(X, Acq_EI_normalized, color='orange', label='EI_normalized')
    axs[1].scatter(X, np.zeros(X.shape[0], ), color='black', marker='X')
    axs[1].set_xlabel('x');
    axs[1].set_ylabel('acq')
    axs[1].legend()

    if plot_cost == True:
        '''cost posterior'''
        u_cost = np.exp(u_latent_cost)

        axs[2].set_title('cost')
        axs[2].scatter(Xt, Yt_cost, color='orange', marker='X')  # evaluated cost points
        axs[2].scatter(xt, ut_cost, color='blue', marker='X')
        axs[2].plot(X, u_cost, color='red', label='cost posterior')  # posterior
        axs[2].fill_between(X[:, 0], np.exp(u_latent_cost[:, 0] + 1.96*sigma_latent_cost[:, 0]),
                            np.exp(u_latent_cost[:, 0] - 1.96*sigma_latent_cost[:, 0]), color='grey', alpha=0.5)

        if plot_true_cost:
            '''true cost'''
            axs[2].set_title('true cost')
            axs[2].plot(X, Y_cost, color='green', label = 'true cost')  # true cost function

        axs[2].legend()
        axs[2].set_xlabel('x');
        axs[2].set_ylabel('cost')

    plt.show()


def plot_posterior_and_acquisitions_for_continuous_domain_noncost(X, Xt, Yt, xt,
               u_X, sigma_X, Acq_dict, model):

    ut, vart= model.predict_f(xt)

    Acq_normalized_dict = {}

    for method in Acq_dict:
        Acq_normalized_dict[method] = Acq_dict[method] / np.sum(Acq_dict[method], axis=0)

    # Acq_normalized= Acq_normalized_dict[list(Acq_normalized_dict.keys())[0]]

    fig, axs = plt.subplots(2)

    '''target posterior'''
    axs[0].set_title('target')
    axs[0].scatter(Xt, Yt, color='orange', marker='X')  # evaluated points
    axs[0].scatter(xt, ut.numpy(), color='blue', marker='X')  # chosen point
    axs[0].plot(X, u_X, color='red', label='target posterior')  # posterior
    axs[0].fill_between(X[:, 0], u_X[:, 0] + sigma_X[:, 0], u_X[:, 0] - sigma_X[:, 0], color='grey', alpha=0.5)
    axs[0].set_xlabel('x');
    axs[0].set_ylabel('y')
    axs[0].legend()

    '''acqusitions'''
    axs[1].set_title('acqusitions')

    for method in Acq_normalized_dict:
        axs[1].plot(X, Acq_normalized_dict[method], label=method)
        # axs[1].plot(X, Acq_normalized, color='green', label= 'EI_cool_normalized')
        # axs[1].plot(X, Acq_EI_pu_normalized, color= 'blue', label= 'EI_pu_normalized')
        # axs[1].plot(X, Acq_EI_normalized, color='orange', label='EI_normalized')
    axs[1].scatter(X, np.zeros(X.shape[0], ), color='black', marker='X')
    axs[1].set_xlabel('x');
    axs[1].set_ylabel('acq')
    axs[1].legend()

    plt.show()

def plot_posterior_and_true_target_cont_domain_2d(X, u_X, Xt, Yt, xt, model, latent_cost_model):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('target')

    ut, vart= model.predict_f(xt); sigmat= np.sqrt(vart)
    ut_latent_cost, vart_latent_cost= latent_cost_model.predict_f(xt); ut_latent_cost= ut_latent_cost.numpy()
    ut_cost = np.exp(ut_latent_cost)

    ax.scatter3D(X[:,0], X[:,1],u_X[:,0], color= 'blue', label= 'posterior target')
    ax.scatter3D(Xt[:,0], Xt[:,1],Yt[:,0],color= 'orange', label= 'evaluated points')
    ax.scatter3D(xt[0,0],xt[1,0], ut_cost[0,0], color= 'blue', label= 'point chosen for eval')


def plot_target_posterior_cont_domain_2d(X, u_X, Xt, Yt, xt, model):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('target posterior, previously evaluated, chosen point')

    ut, vart = model.predict_f(xt);
    sigmat = np.sqrt(vart)

    # ut_latent_cost, vart_latent_cost = latent_cost_model.predict_f(xt);
    # ut_latent_cost = ut_latent_cost.numpy()
    # ut_cost = np.exp(ut_latent_cost)

    ax.scatter3D(X[:, 0], X[:, 1], u_X[:, 0], color='red', label='posterior target')
    ax.scatter3D(Xt[:, 0], Xt[:, 1], Yt[:, 0], color='orange', label='evaluated points')
    ax.scatter3D(xt[0, 0], xt[0,1], ut[0, 0], color='blue', label='point chosen for eval')

    plt.legend()
    plt.show()

def plot_cost_posterior_cont_domain_2d(X, u_cost, Xt, Yt_cost, xt, latent_cost_model):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('cost')

    ut_latent_cost, vart_latent_cost = latent_cost_model.predict_f(xt); ut_latent_cost= ut_latent_cost.numpy()
    ut_cost = np.exp(ut_latent_cost)

    ax.scatter3D(X[:, 0], X[:, 1], u_cost[:, 0], color='red', label='posterior target')
    ax.scatter3D(Xt[:, 0], Xt[:, 1], Yt_cost[:, 0], color='orange', label='evaluated points')
    ax.scatter3D(xt[0, 0], xt[0,1], ut_cost[0, 0], color='blue', label='point chosen for eval')
    ax.legend()
    plt.show()

def plot_acquisitions_cont_domain_colormap_2d(x1_max, x2_max, x1_min, x2_min, disc, Acq_dict, xt, Xt):

    Acq_normalized_dict = {}
    fig, axs = plt.subplots(len(Acq_dict), 1)

    for i, method in enumerate(Acq_dict):
        # Acq_normalized_dict[method] = Acq_dict[method] / np.sum(Acq_dict[method], axis=0)

        # Acq_normalized= Acq_normalized_dict[method]
        Acq_normalized = Acq_dict[method]
        Acq_reshaped= Acq_normalized.reshape(disc,disc)

        print(x1_min, x1_max, x2_min, x2_max)

        axs[i].set_title('Acq_{}'.format(method))
        im = axs[i].imshow(Acq_reshaped, cmap=plt.cm.RdBu, extent=(x1_min, x1_max, x2_max, x2_min))


        # cset = plt.contour(Z, np.arange(-1, 1.5, 0.2), linewidths=2,
        #                    cmap=plt.cm.Set2,
        #                    extent=(-3, 3, -3, 3))
        #
        # plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)

        plt.colorbar(im)
        axs[i].scatter(Xt[:, 0], Xt[:, 1], color='orange')
        axs[i].scatter(xt[0,0], xt[0,1], color= 'blue', label= 'chosen point')
        axs[i].legend()
        axs[i].set_xlabel('x1'); plt.ylabel('x2')
        # plt.show()
    plt.show()

def test_colormap():

    domain= [[-5, 10], [0, 15]]
    disc = 21
    x1_grid = np.linspace(domain[0][0], domain[0][1], disc)
    x2_grid = np.linspace(domain[1][0], domain[1][1], disc)

    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid);

    X1_flat, X2_flat = X1_grid.flatten(), X2_grid.flatten();
    X1_flat, X2_flat = X1_flat.reshape(-1, 1), X2_flat.reshape(-1, 1)
    X_grid = np.append(X1_flat, X2_flat, axis=1)

    Square= (X_grid[:,0]**2+ X_grid[:,1]**2).reshape(-1,1)

    x1_max, x2_max, x1_min, x2_min = np.max(x1_grid), np.max(x2_grid), np.min(x1_grid), np.min(x2_grid)

    Acq_dict = {'Square': Square}

    index_max= int(np.argmax(Square, axis=0))
    x_max= X_grid[index_max, :].reshape(1,-1)
    print('xmax:{}'.format(x_max))
    plot_acquisitions_cont_domain_colormap_2d(x1_max, x2_max, x1_min, x2_min, disc, Acq_dict, x_max)

def compare_posterior_minimium_approximation_with_grid_search(u_X, X, x_pred_opt, model):

    ux, varx = model.predict_f(x_pred_opt)
    plt.figure()
    plt.title('comparison of posterior and posterior minimum approximation')
    plt.plot(X, u_X, color= 'red', label= 'posterior mean')
    plt.scatter(x_pred_opt, ux, color= 'blue', label= 'optimizer app.')
    plt.legend()
    plt.show()

def compare_posterior_minimium_approximation_with_grid_search_2d(u_X, X, x_pred_opt, model):

    ux, varx = model.predict_f(x_pred_opt)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_title('comparison of predicted minimum of posterior with true posterior')

    ax.scatter3D(X[:,0] , X[:,1], u_X[:,0], color= 'red', label= 'posterior')
    ax.scatter3D(x_pred_opt[0,0], x_pred_opt[0,1], ux[0,0], color ='blue', label= 'predicted minimum of posterior')
    plt.legend()
    plt.show()

def plot_acquisition_for_continuous_domain_2d(X, xt, Acq_dict):


    Acq_normalized_dict = {}

    for method in Acq_dict:
        Acq_normalized_dict[method] = Acq_dict[method] / np.sum(Acq_dict[method], axis=0)




    for method in Acq_dict:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_title('Acquisition')

        Acq= Acq_normalized_dict[method]

        ax.scatter3D(X[:,0], X[:,1], Acq[:,0], color= 'blue', label= method)
        ax.scatter3D(xt[0,0], xt[0,1], 0, color= 'red', label= 'estimated optimum of acquisition')

        plt.legend()
        plt.show()

def drawArrow(A, B):

    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1],
              head_width=3, length_includes_head=True)

def plot_colormaps(Xt, disc, objective_func, cost_function, domain, name):


    x1_min= domain[0][0]; x1_max= domain[0][1]; x2_min= domain[0][0]; x2_max= domain[0][1];
    X_grid= create_grid(disc, domain)
    true_objective= objective_func(X_grid); true_objective = true_objective.reshape(disc, disc)
    true_cost= cost_function(X_grid); true_cost = true_cost.reshape(disc, disc)

    plt.figure()
    plt.title('{}:True objective'.format(name))
    im = plt.imshow(true_objective, cmap=plt.cm.RdBu, extent=(x1_min, x1_max, x2_max, x2_min))
    plt.colorbar(im)
    plt.scatter(Xt[:,0], Xt[:,1], color= 'orange', label= 'evaluation points')
    plt.scatter(Xt[0, 0], Xt[0, 1], color='red')
    plt.scatter(Xt[-1, 0], Xt[-1, 1], color='black')

    for i in range(Xt.shape[0]):
        plt.text(Xt[i,0], Xt[i, 1], '{}'.format(i), color= 'black')
    # for i in range(Xt.shape[0]):
    #     if i<Xt.shape[0]-1:
    #         plt.arrow(Xt[i, 0], Xt[i, 1], (Xt[i+1, 0] - Xt[i, 0]), (Xt[i+1, 1] - Xt[i, 1]), head_width=0.04,
    #                   length_includes_head=True, color= 'orange' )
    plt.legend()
    plt.xlabel('x1');
    plt.ylabel('x2')
    plt.show()

    plt.figure()
    plt.title('{}:True cost'.format(name))
    im = plt.imshow(true_cost, cmap=plt.cm.RdBu, extent=(x1_min, x1_max, x2_max, x2_min))
    plt.colorbar(im)

    plt.scatter(Xt[:,0], Xt[:,1], color= 'orange', label= 'evaluation points')
    plt.scatter(Xt[0, 0], Xt[0, 1], color='red')
    plt.scatter(Xt[-1, 0], Xt[-1, 1], color='black')
    for i in range(Xt.shape[0]):
        plt.text(Xt[i,0], Xt[i, 1], '{}'.format(i), color= 'black')
    # for i in range(Xt.shape[0]):
    #     if i<Xt.shape[0]-1:
    #         plt.arrow(Xt[i, 0], Xt[i, 1], (Xt[i+1, 0] - Xt[i, 0]), (Xt[i+1, 1] - Xt[i, 1]), head_width=0.04,
    #                   length_includes_head=True, color= 'orange' )
    plt.legend()
    plt.xlabel('x1');
    plt.ylabel('x2')
    plt.show()

def plot_evaluation_arrows_1d(Xt, Yt, disc, objective_func, cost_function, domain, name):

    X_grid= create_grid(disc, domain)
    true_objective= objective_func(X_grid);
    true_cost= cost_function(X_grid)

    plt.figure()
    plt.title('{}:True objective'.format(name))
    plt.plot(X_grid[:, 0], true_objective[:,0], label= 'true objective', color= 'green')

    plt.scatter(Xt[:,0], Yt[:,0],  color= 'orange', label= 'evaluation points')

    for i in range(Xt.shape[0]):
        plt.text(Xt[i,0], Yt[i, 0], '{}'.format(i))

    plt.legend()
    plt.xlabel('x');
    plt.ylabel('y')
    plt.show()
    #
    # plt.figure()
    # plt.title('True cost')
    #
    # plt.scatter(Xt[:,0], Yt_cost[:,1], color= 'orange', label= 'evaluation points')
    # plt.plot(X_grid[:, 0], true_cost[:,0], label= 'true cost', color= 'green')
    #
    # for i in range(Xt.shape[0]):
    #     plt.text(Xt[i,0], Yt_cost[i, 0], '{}'.format(i))
    #
    # plt.legend()
    # plt.xlabel('x');
    # plt.ylabel('y')
    # plt.show()

def plot_contour_and_save(objective_func, domain, disc, name):
    d1= 0.01; d2= 0.01
    X = np.arange(domain[0][0], domain[0][1], d1)
    n1= int((domain[0][1]- domain[0][0])/d1)
    Y = np.arange(domain[1][0], domain[1][1], d2)
    n2 = int((domain[1][1] - domain[1][0]) / d2)

    X1_grid, X2_grid = np.meshgrid(X, Y)
    X1_grid_flat, X2_grid_flat= (X1_grid.flatten()).reshape(-1,1), (X2_grid.flatten()).reshape(-1,1)
    X= np.append(X1_grid_flat, X2_grid_flat, axis=1)

    Y = objective_func(X)
    Y= Y.reshape(n1,n2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    print(X.shape)
    ax.contour3D(X1_grid, X2_grid, Y, 50, cmap='binary')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('value');
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.family'] = 'Calibri'
    # plt.legend()

    plt.savefig('../Results/'+name+'.pdf', dpi=400)
    # plt.show()

