import numpy as np


models = ['node', 'sonode_conv_v', 'anode']
experiment_numbers = ['1', '2', '3']
array_types = ['train_acc', 'test_acc']



def make_moving_av(model_no, experiment_no, array_type_no):

    model = models[model_no]
    experiment_no = experiment_numbers[experiment_no]
    array_type = array_types[array_type_no]
    
    filename = 'experiment_'+model+experiment_no+'./'
    
    accuracy = np.load(filename+array_type+'_arr.npy')
    samp_eps_array = np.load(filename+'epoch_arr.npy')
    
    
    window = 5
    def moving_average(a, periods=window):
        weights = np.ones(periods) / periods
        return np.convolve(a, weights, mode='valid')
    
    
    accuracy_ma = moving_average(accuracy)
    samp_eps_array = samp_eps_array[:len(samp_eps_array)-window+1]
    np.save(filename+'running_'+array_type+'.npy', accuracy_ma)
    np.save(filename+'running_epoch_arr.npy', samp_eps_array)


for i in range(3):
    for j in range(3):
        for k in range(2):
            make_moving_av(i, j, k)

