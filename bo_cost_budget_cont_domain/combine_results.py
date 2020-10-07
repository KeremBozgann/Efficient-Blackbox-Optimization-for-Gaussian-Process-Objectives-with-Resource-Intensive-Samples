import h5py
import numpy as np

from util_results import *
from plots import plot_and_save_average_loss_and_std

def get_std_list(loss_and_count_dict):
    new_dict= {}
    for method in loss_and_count_dict:
        count= loss_and_count_dict[method][1]; loss= loss_and_count_dict[method][0]
        cost_grid= loss_and_count_dict[method][2]
        f_best= loss_and_count_dict[method][3]
        loss_list= loss_and_count_dict[method][4]
        count_list= loss_and_count_dict[method][5]

        '''get std for method'''
        std_list= np.zeros([1, loss_list.shape[1]])
        for j in range(loss_list.shape[1]):
            loss_listj= loss_list[:, j]
            count_listj= count_list[:, j]
            loss_listj_valid= loss_listj[np.where(count_listj!=0)]
            if len(loss_listj_valid!=0):
                std_list[0, j]= np.std(loss_listj_valid)
            else:
                std_list[0, j]= -1

        new_dict[method]= [loss,count,cost_grid, f_best, loss_list, count_list, std_list]

    return new_dict

def delete_invalid_loss_and_count(loss_and_count_dict):

    print('entered delete invalid loss')
    new_dict= {}
    for method in loss_and_count_dict:
        count= loss_and_count_dict[method][1]; loss= loss_and_count_dict[method][0]
        cost_grid= loss_and_count_dict[method][2]
        f_best= loss_and_count_dict[method][3]
        loss_list= loss_and_count_dict[method][4]
        count_list= loss_and_count_dict[method][5]
        std_list= loss_and_count_dict[method][6]
        print('count', count)
        print('np.where(count==0)', np.where(count==0))
        invalid= np.where(count==0)[0]
        print('invalid:', invalid)
        loss= np.delete(loss, invalid)
        count= np.delete(count, invalid)
        cost_grid= np.delete(cost_grid, invalid)
        f_best= np.delete(f_best, invalid)
        std_list= np.delete(std_list, invalid)
        new_dict[method]= [loss,count,cost_grid, f_best, loss_list, count_list, std_list]

    return new_dict

folder= '14_09_2020'
file_dir1= '../Results/'+folder+'/exp1.h5'
files= [file_dir1]

num_iter1= 10
methods= ['ei_pu', 'carbo', 'ei',  'eiw_eipu']
cost_disc_num= 100

folder= '14_09_2020'
exp_name= 'exp2'


'''initialize loss and count dict'''

loss_and_count_dict= {}
loss_list= np.empty([0, cost_disc_num])
count_list= np.empty([0, cost_disc_num])

for method in methods:
    method_list= []
    for i in range(4):
        method_list.append(np.zeros([cost_disc_num]))
    method_list.append(loss_list)
    method_list.append(count_list)

    loss_and_count_dict[method]= method_list

for file in files:

    '''add first file results'''
    with h5py.File(file, 'r') as hf:


        for method in methods:

            count = np.zeros([cost_disc_num])
            loss = np.zeros([cost_disc_num])
            f_best = np.zeros([cost_disc_num])
            loss_list= np.zeros([0, cost_disc_num])
            count_list= np.zeros([0, cost_disc_num])
            for iter_num in range(num_iter1):
                # print(np.array(hf.get('iter_{}_method_{}_loss'.format(iter_num, method))))
                lossi= np.array(hf.get('iter_{}_method_{}_loss'.format(iter_num, method)))
                counti= np.array(hf.get('iter_{}_method_{}_count'.format(iter_num, method)))
                costi = np.array(hf.get('iter_{}_method_{}_cost_grid'.format(iter_num, method)))
                f_besti= np.array(hf.get('iter_{}_method_{}_f_best'.format(iter_num, method)))

                loss+= lossi
                count+= counti
                f_best+= f_besti

                loss_list= np.append(loss_list, np.atleast_2d(lossi), axis=0)
                count_list =  np.append(count_list, np.atleast_2d(counti), axis=0)

            method_list = loss_and_count_dict[method]

            method_list[0] += loss
            method_list[1] += count
            method_list[2] = costi
            method_list[3] += f_best
            method_list[4]= np.append(method_list[4], loss_list, axis=0)
            method_list[5]= np.append(method_list[5], count_list, axis=0)
            loss_and_count_dict[method]= method_list


loss_and_count_dict= get_std_list(loss_and_count_dict)
loss_and_count_dict= delete_invalid_loss_and_count(loss_and_count_dict)

plot_and_save_average_loss_and_std(loss_and_count_dict, folder, exp_name)
#
# save_results_without_params(loss_and_count_dict, folder, exp_name)