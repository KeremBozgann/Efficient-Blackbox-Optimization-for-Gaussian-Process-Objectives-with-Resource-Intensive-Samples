import h5py
import numpy as np

from util_results import *
from plots import plot_and_save_average_loss

def delete_invalid_loss_and_count(loss_and_count_dict):

    print('entered delete invalid loss')
    new_dict= {}
    for method in loss_and_count_dict:
        count= loss_and_count_dict[method][1]; loss= loss_and_count_dict[method][0]
        cost_grid= loss_and_count_dict[method][2]
        f_best= loss_and_count_dict[method][3]
        print('count', count)
        print('np.where(count==0)', np.where(count==0))
        invalid= np.where(count==0)[0]
        print('invalid:', invalid)
        loss= np.delete(loss, invalid)
        count= np.delete(count, invalid)
        cost_grid= np.delete(cost_grid, invalid)
        f_best= np.delete(f_best, invalid)
        new_dict[method]= [loss,count,cost_grid, f_best]

    return new_dict

folder= '14_09_2020'
file_dir1= '../Results/'+folder+'/exp1.h5'
files= [file_dir1]

num_iter1= 7
methods= ['ei_pu', 'carbo', 'ei',  'imco']
cost_disc_num= 100

folder= '14_09_2020'
exp_name= 'exp2'


'''initialize loss and count dict'''

loss_and_count_dict= {}

for method in methods:
    method_list= []
    for i in range(4):
        method_list.append(np.zeros([cost_disc_num]))

    loss_and_count_dict[method]= method_list

for file in files:

    '''add first file results'''
    with h5py.File(file, 'r') as hf:


        for method in methods:

            count = np.zeros([cost_disc_num])
            loss = np.zeros([cost_disc_num])
            f_best = np.zeros([cost_disc_num])

            for iter_num in range(num_iter1):
                # print(np.array(hf.get('iter_{}_method_{}_loss'.format(iter_num, method))))
                lossi= np.array(hf.get('iter_{}_method_{}_loss'.format(iter_num, method)))
                counti= np.array(hf.get('iter_{}_method_{}_count'.format(iter_num, method)))
                costi = np.array(hf.get('iter_{}_method_{}_cost_grid'.format(iter_num, method)))
                f_besti= np.array(hf.get('iter_{}_method_{}_f_best'.format(iter_num, method)))

                loss+= lossi
                count+= counti
                f_best+= f_besti

            method_list = loss_and_count_dict[method]

            method_list[0] += loss
            method_list[1] += count
            method_list[2] = costi
            method_list[3] += f_best
            loss_and_count_dict[method]= method_list



loss_and_count_dict= delete_invalid_loss_and_count(loss_and_count_dict)

plot_and_save_average_loss(loss_and_count_dict, folder, exp_name)
#
# save_results_without_params(loss_and_count_dict, folder, exp_name)