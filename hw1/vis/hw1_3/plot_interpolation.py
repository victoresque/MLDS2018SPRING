import argparse, os, sys
import numpy as np
import torch
from collections import OrderedDict

from torch.autograd import Variable
from models.loss import cross_entropy_loss
from models.models import DeepMnistCNN
from models.metric import accuracy
from data_loader.data_loader import MnistLoader
from utils.util import split_validation

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def main():
    parser = argparse.ArgumentParser(description='hw1-3-3 interpolation plot')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        help='batch size to computing loss (default: 128)')
    parser.add_argument('--sample', default=10, type=int,
                        help='the number of the sample around each epoch (default: 10)')
    parser.add_argument('--task', default='batch', type=str, help='task to deal with [batch, lr] (default: batch)')
    parser.add_argument('--folder-0', type=str, help='model 0\' file name (example: batch64)')
    parser.add_argument('--folder-1', type=str, help='model 1\' file name (example: batch1024)')
    parser.add_argument('--cuda', action='store_true', help='use gpu')
    parser.add_argument('--log', action='store_true', help='loss plot in log scale')
    args = parser.parse_args()

    state = {}
    base = 'models/saved/1-3-3/part1/'
    model_name = 'DeepMnistCNN'
    plot_method = 'semilogy' if args.log else 'plot'
    loss = cross_entropy_loss
    data_loader = MnistLoader(args.batch)
    data_loader, valid_data_loader = split_validation(data_loader, validation_split=0.1, randomized=False)
    
    folder_0_path = os.path.join(base, args.folder_0)
    folder_1_path = os.path.join(base, args.folder_1)

    files = os.listdir(folder_0_path)
    files.sort(key=lambda s: int(s[s.find('epoch')+5: s.rfind('_loss')]))
    checkpoint_0_path = os.path.join(folder_0_path, files[-1]) 

    files = os.listdir(folder_1_path)
    files.sort(key=lambda s: int(s[s.find('epoch')+5: s.rfind('_loss')]))
    checkpoint_1_path = os.path.join(folder_1_path, files[-1]) 

    checkpoint_0 = torch.load(checkpoint_0_path)
    checkpoint_1 = torch.load(checkpoint_1_path)
    checkpoint_epoch = (checkpoint_0['epoch'], checkpoint_1['epoch'])

    state[0] = {
                'state_dict': checkpoint_0['state_dict'],
                'epoch':      checkpoint_epoch[0],
                'acc':        checkpoint_0['logger'].entries[checkpoint_epoch[0]]['accuracy'],
                'val_acc':    checkpoint_0['logger'].entries[checkpoint_epoch[0]]['val_accuracy'],
                'loss':       checkpoint_0['logger'].entries[checkpoint_epoch[0]]['loss'],
                'val_loss':   checkpoint_0['logger'].entries[checkpoint_epoch[0]]['val_loss']
                }

    state[1] = {
                'state_dict': checkpoint_1['state_dict'],
                'epoch':      checkpoint_epoch[1],
                'acc':        checkpoint_1['logger'].entries[checkpoint_epoch[1]]['accuracy'],
                'val_acc':    checkpoint_1['logger'].entries[checkpoint_epoch[1]]['val_accuracy'],
                'loss':       checkpoint_1['logger'].entries[checkpoint_epoch[1]]['loss'],
                'val_loss':   checkpoint_1['logger'].entries[checkpoint_epoch[1]]['val_loss']
                }
    
    checkpoint_weight_vectors = []
    checkpoint_acc = []
    checkpoint_val_acc = []
    checkpoint_loss = []
    checkpoint_val_loss = []
    for i in range(2):
        checkpoint_weight_vectors.append(orderdict_flatten(state[i]['state_dict']))
        checkpoint_acc.append(state[i]['acc'])
        checkpoint_val_acc.append(state[i]['val_acc'])
        checkpoint_loss.append(state[i]['loss'])
        checkpoint_val_loss.append(state[i]['val_loss'])

    checkpoint_weight_vectors = np.array(checkpoint_weight_vectors)
    weight_length = model_weight_length(model_name)
    try:
        assert(checkpoint_weight_vectors.shape[1] == weight_length)
    except:
        print("ASSERTION ERROR assert(checkpoint_weight_vectors.shape[1] == weight_length)")
        print(checkpoint_weight_vectors.shape, weight_length)
        return
    
    # interpolation loss
    interp_epoch, interp_features = generate_iterpolation_features(checkpoint_weight_vectors, model_name, args,
                                                                   data_loader, valid_data_loader, loss, state)
    interp_loss = interp_features[0]
    interp_acc = interp_features[1]
    interp_val_loss = interp_features[2]
    interp_val_acc = interp_features[3]

    # plot the figure
    plt.figure(figsize=(20, 10))

    fig, ax1 = plt.subplots()
    ln1 = eval('ax1.' + plot_method)(interp_epoch, interp_loss, 'b-', label='train loss')
    ln2 = eval('ax1.' + plot_method)(interp_epoch, interp_val_loss, 'b--', label='valid loss')
    ax1.set_xlabel('alpha')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('cross entropy', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ln3 = ax2.plot(interp_epoch, interp_acc, 'r-', label='train acc')
    ln4 = ax2.plot(interp_epoch, interp_val_acc, 'r--', label='valid acc')
    ax2.set_ylabel('accuracy', color='r')
    ax2.tick_params('y', colors='r')

    lns = ln1+ln2+ln3+ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.set_title('MNIST learning rate 1e-3 vs. 1e-2')
    
    fig.tight_layout()
    if args.log:
        savefig_path = './{}_inerpolation_sample{}_log.png'.format(args.task, args.sample)
    else:
        savefig_path = './{}_inerpolation_sample{}.png'.format(args.task, args.sample)
    fig.savefig(savefig_path)
    print("saving file : {}".format(savefig_path))

# From state_dict to 1-D np.array
def orderdict_flatten(orderdict):
    flat_vec = []
    for key, value in orderdict.items():
        sz = np.array(value.size())
        sz_idx = sz - 1
        cum_idx = np.zeros(len(sz)).astype(np.int16)
        while cum_idx[0] < sz[0]:
            flat_vec.append(value[tuple(cum_idx)])
            cum_idx[-1] += 1
            for i in range(len(sz)-1, 0, -1):
                if cum_idx[i] == sz[i]:
                    cum_idx[i-1] += 1
                    cum_idx[i] = 0

    return np.array(flat_vec)


# From 1-D np.array to state_dict
def generate_state_dict(vector, model_name):
    model = eval(model_name)()
    return_orderdict = OrderedDict()
    dummy_state_dict = model.state_dict()
    cum_idx = 0
    for key, value in dummy_state_dict.items():
        sz = np.array(value.size())
        length = np.prod(sz)
        weight = vector[cum_idx:cum_idx + length].reshape(sz)
        return_orderdict[key] = torch.FloatTensor(weight)
        cum_idx = cum_idx + length
    return return_orderdict


def model_weight_length(model_name):
    length = 0
    dummy_model = eval(model_name)()
    for key, value in dummy_model.state_dict().items():
        length += np.prod(np.array(value.size()))
    return(length)


def generate_iterpolation_features(checkpoint_weight_vectors, model_name, args, data_loader, valid_data_loader, loss, state):
    n_sample = args.sample
    all_weight_vectors = []
    print(" Start Interpolation Sampling ... ")
    sample_weight_vec_n1 = 2*checkpoint_weight_vectors[0] - checkpoint_weight_vectors[1]
    sample_weight_vec_p2 = 2*checkpoint_weight_vectors[1] - checkpoint_weight_vectors[0]

    all_weight_vectors.append(sample_weight_vec_n1)
    # alpha (-1, 0)
    for j in range(1, n_sample):
        tmp_vector = (checkpoint_weight_vectors[0] * j + sample_weight_vec_n1 * (n_sample - j)) / n_sample
        all_weight_vectors.append(tmp_vector)
    all_weight_vectors.append(checkpoint_weight_vectors[0])
    # alpha (0, 1)
    for j in range(1, n_sample):
        tmp_vector = (checkpoint_weight_vectors[1] * j + checkpoint_weight_vectors[0] * (n_sample - j)) / n_sample
        all_weight_vectors.append(tmp_vector)
    all_weight_vectors.append(checkpoint_weight_vectors[1])
    # alpha (1, 2)
    for j in range(1, n_sample):
        tmp_vector = (sample_weight_vec_p2 * j + checkpoint_weight_vectors[1] * (n_sample - j)) / n_sample
        all_weight_vectors.append(tmp_vector)
    all_weight_vectors.append(sample_weight_vec_p2)
    all_weight_vectors = np.array(all_weight_vectors)
    print(" Interpolation Sampling Completed \n")

    all_weight_vectors_shape = all_weight_vectors.shape

    print(" Computing Interpolation training set loss and acc...")
    all_loss, all_acc = [], []
    for i in range(all_weight_vectors_shape[0]):
        if i == n_sample:
            all_loss.append(state[0]['loss'])
            all_acc.append(state[0]['acc'])
            continue
        if i == 2*n_sample:
            all_loss.append(state[1]['loss'])
            all_acc.append(state[1]['acc'])
            continue
            
        test_model = eval(model_name)()
        vector_state_dict = generate_state_dict(all_weight_vectors[i], model_name)
        test_model.load_state_dict(vector_state_dict)

        if args.cuda: test_model.cuda()

        sys.stdout.write('\b'*100)
        sys.stdout.flush()

        total_loss, total_acc = 0,0
        for batch_idx, (data, target) in enumerate(data_loader):
            target_dtype = str(target.dtype)
            data = torch.FloatTensor(data)
            target = torch.FloatTensor(target) if target_dtype[0] == 'f' else torch.LongTensor(target)
            data, target = Variable(data), Variable(target)

            if args.cuda: data, target = data.cuda(), target.cuda()

            output = test_model(data)
            batch_loss = loss(output, target)
            total_loss += batch_loss.data[0]

            y_output = output.data.cpu().numpy()
            y_output = np.argmax(y_output, axis=1)
            y_target = target.data.cpu().numpy()
            total_acc += accuracy(y_output, y_target)

        avg_loss = total_loss / len(data_loader)
        avg_acc = total_acc / len(data_loader)
        all_loss.append(avg_loss)
        all_acc.append(avg_acc)

        sys.stdout.write("({}) loss = {}, acc = {}".format(i+1, avg_loss, avg_acc))
        sys.stdout.flush()
    print('\n')


    print(" Computing Interpolation validation set loss and acc...")
    all_val_loss, all_val_acc = [], []
    for i in range(all_weight_vectors_shape[0]):
        if i == n_sample:
            all_val_loss.append(state[0]['val_loss'])
            all_val_acc.append(state[0]['val_acc'])
            continue
        if i == 2*n_sample:
            all_val_loss.append(state[1]['val_loss'])
            all_val_acc.append(state[1]['val_acc'])
            continue
            
        test_model = eval(model_name)()
        vector_state_dict = generate_state_dict(all_weight_vectors[i], model_name)
        test_model.load_state_dict(vector_state_dict)

        if args.cuda: test_model.cuda()

        sys.stdout.write('\b'*100)
        sys.stdout.flush()

        total_val_loss, total_val_acc = 0,0
        for batch_idx, (data, target) in enumerate(valid_data_loader):
            target_dtype = str(target.dtype)
            data = torch.FloatTensor(data)
            target = torch.FloatTensor(target) if target_dtype[0] == 'f' else torch.LongTensor(target)
            data, target = Variable(data), Variable(target)

            if args.cuda: data, target = data.cuda(), target.cuda()

            output = test_model(data)
            batch_loss = loss(output, target)
            total_val_loss += batch_loss.data[0]

            y_output = output.data.cpu().numpy()
            y_output = np.argmax(y_output, axis=1)
            y_target = target.data.cpu().numpy()
            total_val_acc += accuracy(y_output, y_target)

        avg_val_loss = total_val_loss / len(valid_data_loader)
        avg_val_acc = total_val_acc / len(valid_data_loader)
        all_val_loss.append(avg_val_loss)
        all_val_acc.append(avg_val_acc)

        sys.stdout.write("({}) val_loss = {}, val_acc = {}".format(i+1, avg_val_loss, avg_val_acc))
        sys.stdout.flush()
    print('\n')

    return np.linspace(-1, 2, 3*n_sample+1), (np.array(all_loss), np.array(all_acc), np.array(all_val_loss), np.array(all_val_acc))

if __name__ == '__main__':
    main()
