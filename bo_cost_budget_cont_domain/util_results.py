
import numpy as np
import h5py

def save_to_hf(loss_and_count_dict, folder, hf_name, iter_num):


    for method in loss_and_count_dict:

        [loss, count, cost_grid, f_best]= loss_and_count_dict[method]

        with h5py.File('../Results/'+ folder+ '/'+ hf_name+'.h5', 'a') as hf:

            hf.create_dataset('iter_{}_method_{}_loss'.format(iter_num, method), data= loss)
            hf.create_dataset('iter_{}_method_{}_count'.format(iter_num, method), data= count)
            hf.create_dataset('iter_{}_method_{}_cost_grid'.format(iter_num, method), data= cost_grid)
            hf.create_dataset('iter_{}_method_{}_f_best'.format(iter_num, method), data= f_best)


def save_results(loss_and_count_dict, parameter_dict, folder, exp_name):

    for method in loss_and_count_dict:

        params_of_method= parameter_dict[method]
        [loss, count, cost_grid, f_best]= loss_and_count_dict[method]

        with open('../Results/'+folder+ '/'+exp_name+'.txt', 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write(method+':\n')
            f.write('-----\n')
            f.write('Parameters:\n')
            f.write('\n')
            for param_name in params_of_method:
                f.write(param_name+ str(params_of_method[param_name])+'\n')
            f.write('\n')
            f.write('\n')

            f.write('loss: '+str(loss)+'\n')
            f.write('count: '+str(count)+'\n')
            f.write('cost_grid: '+str(cost_grid)+'\n')
            f.write('f_best_list:' + str(f_best)+'\n')
            f.write('\n')
            f.write('\n')

def save_results_without_params(loss_and_count_dict, folder, exp_name):

    for method in loss_and_count_dict:

        [loss, count, cost_grid, f_best]= loss_and_count_dict[method]

        with open('../Results/'+folder+ '/'+exp_name+'.txt', 'a') as f:
            f.write('\n')
            f.write('\n')
            f.write(method+':\n')
            f.write('-----\n')
            f.write('\n')
            f.write('\n')

            f.write('loss: '+str(loss)+'\n')
            f.write('count: '+str(count)+'\n')
            f.write('cost_grid: '+str(cost_grid)+'\n')
            f.write('f_best_list:' + str(f_best)+'\n')
            f.write('\n')
            f.write('\n')

def find_cell(cost, cost_grid):

    index_list= np.where(cost<=cost_grid)[0]

    if not len(index_list)==0:
        index= index_list[0]
        return index

    else:
        return -1

def add_loss(loss, count, loss_list, cum_cost_list, f_best_list, f_best, cost_grid):

    for i in range(len(cum_cost_list)):

        index= find_cell(cum_cost_list[i], cost_grid)

        if not index==-1:
            loss[index]+= loss_list[i]
            count[index]+= 1
            f_best[index]+= f_best_list[i]

    return loss, count, f_best


def delete_invalid_loss_and_count(loss_and_count_dict):

    new_dict= {}
    for method in loss_and_count_dict:
        count= loss_and_count_dict[method][1]; loss= loss_and_count_dict[method][0]
        cost_grid= loss_and_count_dict[method][2]
        f_best= loss_and_count_dict[method][3]
        invalid= np.where(count==0)[0]

        loss= np.delete(loss, invalid)
        count= np.delete(count, invalid)
        cost_grid= np.delete(cost_grid, invalid)
        f_best= np.delete(f_best, invalid)
        new_dict[method]= [loss,count,cost_grid, f_best]

    return new_dict