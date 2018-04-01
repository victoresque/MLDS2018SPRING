import numpy as np
import torch
from torch.autograd import Variable
from models.loss import mse_loss, cross_entropy_loss
from models.models import DeepFC, MiddleFC, ShallowFC
from models.models import DeepMnistCNN, MiddleMnistCNN, ShallowMnistCNN
from models.models import DeepCifarCNN, MiddleCifarCNN, ShallowCifarCNN
from data_loader.function_data_loader import FunctionDataLoader
from data_loader.data_loader import MnistLoader, CifarLoader
from MulticoreTSNE import MulticoreTSNE as TSNE
from collections import OrderedDict
import argparse, os, sys

import matplotlib
matplotlib.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def main():
    parser = argparse.ArgumentParser(description='plot error surface')
    parser.add_argument('-b', '--batch', default=128, type=int,
                        help='batch size to computing loss (default: 128)')
    parser.add_argument('--arch', default='deep', type=str,
                        help='arch [deep, middle, shallow] (default: deep)')
    parser.add_argument('--dataset', default='stair', type=str,
                        help='choose the dataset for plotting [mnist, cifar, sinc, stair, gibbs, sqwave, sumsin] (default: stair)')
    parser.add_argument('--sample', default=10, type=int,
                        help='the number of the sample around each epoch (default: 10)')
    parser.add_argument('--min-loss', default=2000, type=float,
                        help='upper bound when calculate loss (default: 2000)')
    parser.add_argument('--cuda', action='store_true', help='use gpu')

    args = parser.parse_args()

    """
    state = {
                1: {
                     'state_dict': < OrderedDict >,
                     'loss':       < float >,
                     'epoch':
                   }
            }

    """
    state = {}
    base = './models/saved/1-2-b/' + args.arch.title() + args.dataset.title()
    if  args.dataset == 'mnist' or args.dataset == 'cifar':
        model_name = args.arch.title() + args.dataset.title() + 'CNN'
        loss = cross_entropy_loss
        data_loader = eval(args.dataset.title() + 'Loader')(args.batch)
    else:
        model_name = args.arch.title() + 'FC'
        loss = mse_loss
        data_loader = FunctionDataLoader(args.dataset,
                                         batch_size=args.batch,
                                         n_sample=20000, x_range=(0, 1))

    
    if not os.path.exists(base) or len(os.listdir(base)) <= 0:
        print("Please train the according dataset in advanced!")
        print("Run command : ")
        if args.dataset == 'mnist' or args.dataset == 'cifar':
            print(" python3 main.py -e 10000 --save-freq 100 --save-dir models/saved/1-2-b/{}{} --arch {} --dataset {} \
                ".format(args.arch.title(), args.dataset.title(), args.arch, args.dataset))
        else:
            print(" python3 main.py -e 10000 --save-freq 100 --save-dir models/saved/1-2-b/{}{} --arch {} --target-func {} \
                ".format(args.arch.title(), args.dataset.title(), args.arch, args.dataset))
        return
    
    checkpoint_epochs = []
    for checkpoint_filenames in os.listdir(base):
        checkpoint_path = os.path.join(base, checkpoint_filenames)
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        state[epoch] = {'state_dict': checkpoint['state_dict'],
                        'epoch': epoch }
        checkpoint_epochs.append(epoch)
    checkpoint_epochs.sort()

    # shape = (epochs, weight_length)
    checkpoint_weight_vectors = []
    for epoch in checkpoint_epochs:
        checkpoint_weight_vectors.append(orderdict_flatten(state[epoch]['state_dict']))
    checkpoint_weight_vectors = np.array(checkpoint_weight_vectors)
    std_dev_at_dim = np.std(checkpoint_weight_vectors, axis=0)
    weight_length = model_weight_length(model_name)
    assert(std_dev_at_dim.shape[0] == weight_length)
    

    # shape = (epochs*n_sample, weight_length)
    sample_weight_vectors = []
    print(" Start Sampling ... ")
    for i in range(len(checkpoint_epochs)):
        for j in range(args.sample):
            tmp_vector = np.random.randn(weight_length) * std_dev_at_dim + checkpoint_weight_vectors[i]
            sample_weight_vectors.append(tmp_vector)
    sample_weight_vectors = np.array(sample_weight_vectors)
    print(" Sampling Completed \n")
                
    all_weight_vectors = np.append(checkpoint_weight_vectors, sample_weight_vectors, axis=0)           
    print(" Starting Computing TSNE ... ")
    tsne = TSNE(n_components=2, n_jobs=4, verbose=1, random_state=0)
    tsne_projection = tsne.fit_transform(all_weight_vectors)
    print(" TSNE Completed \n")
            
    print(" Computing Sample Loss ... ")
    all_weight_vectors_shape = all_weight_vectors.shape
    all_loss = []
    for i in range(all_weight_vectors_shape[0]):
        test_model = eval(model_name)()
        vector_state_dict = generate_state_dict(all_weight_vectors[i], model_name)
        test_model.load_state_dict(vector_state_dict)
        
        if args.cuda: test_model.cuda()
        
        sys.stdout.write('\b'*50)
        sys.stdout.flush()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            target_dtype = str(target.dtype)
            data = torch.FloatTensor(data)
            target = torch.FloatTensor(target) if target_dtype[0] == 'f' else torch.LongTensor(target)
            data, target = Variable(data), Variable(target)

            if args.cuda: data, target = data.cuda(), target.cuda()

            output = test_model(data)
            batch_loss = loss(output, target)
            total_loss += batch_loss.data[0]

        avg_loss = total_loss / len(data_loader)
        if avg_loss > args.min_loss:
            all_loss.append(args.min_loss)
        else:
            all_loss.append(avg_loss)
        sys.stdout.write("({}) loss = {}".format(i+1, avg_loss))
        sys.stdout.flush()
    print('')
    tsne_projection = np.array(tsne_projection)
    all_loss = np.array(all_loss)
    assert(tsne_projection.shape[0] == all_loss.shape[0])
    
    # plot error surface
    x = tsne_projection[:,0]
    y = tsne_projection[:,1]
    z = all_loss
    
    z_min = np.min(z)
    z_max = np.max(z)
    c = (z + z_min) / (z_max - z_min)
    cmap = cm.get_cmap('coolwarm')
    
    checkpoint_length = len(checkpoint_epochs)
    x_check, x_sample = x[:checkpoint_length], x[checkpoint_length:]
    y_check, y_sample = y[:checkpoint_length], y[checkpoint_length:]
    z_check, z_sample = z[:checkpoint_length], z[checkpoint_length:]
    c_check, c_sample = c[:checkpoint_length], c[checkpoint_length:]
    
    # interpolation loss
    interp_epoch, interp_loss = generate_iterpolation_loss(checkpoint_epochs, checkpoint_weight_vectors,
                                                           model_name, args, data_loader, loss)
    
    
    fig = plt.figure(figsize=(24,12))
        
    # subplot 1
    ax1 = fig.add_subplot(2, 3, 2, projection='3d')
    ax1.plot_trisurf(x, y, np.log10(z), linewidth=0.2, antialiased=True, alpha=0.8)
    ax1.plot(x_check, y_check, np.log10(z_check), color='purple', linewidth=3, alpha=1)
    ax1.set_zticks([])
    ax1.set_title('error surface # loss(z) in log scale')
    
    # subplot 2
    ax2 = fig.add_subplot(2, 3, 1, projection='3d')
    p2 = ax2.scatter(x, y, np.log10(z), c=z, cmap=cmap, alpha=0.6)
    ax2.plot(x_check, y_check, np.log10(z_check), color='purple', linewidth=3, alpha=1)
    ax2.set_zticks([])
    ax2.set_title('loss sample # loss(z) in log scale')
    fig.colorbar(p2)
    
    # subplot 3
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.semilogy(interp_epoch, interp_loss)
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('loss')
    ax3.set_title('interpolation loss')
    ax3.grid()
    
    # subplot 4
    ax4 = fig.add_subplot(2, 3, 5, projection='3d')
    ax4.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, alpha=0.8)
    ax4.plot(x_check, y_check, z_check, color='purple', linewidth=3, alpha=1)
    ax4.set_title('error surface')
    
    # subplot 5
    ax5 = fig.add_subplot(2, 3, 4, projection='3d')
    p5 = ax5.scatter(x, y, z, c=z, cmap=cmap, alpha=0.6)
    ax5.plot(x_check, y_check, z_check, color='purple', linewidth=3, alpha=1)
    ax5.set_title('loss sample')
    fig.colorbar(p5)
        
    # subplot 6
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(interp_epoch, interp_loss)
    ax6.set_xlabel('epochs')
    ax6.set_ylabel('loss')
    ax6.set_title('interpolation loss')
    ax6.grid()
    
    plt.tight_layout()
    save_file_path = './{}{}_error_surface_sample{}.png'.format(args.arch, args.dataset, args.sample)
    plt.savefig(save_file_path)
    print("Saving file : {}".format(save_file_path))

    
    
# From state_dict to 1-D np.array
def orderdict_flatten(orderdict):
    flat_vec = []
    for key, value in orderdict.items():
        sz = np.array(value.size())
        if len(sz) == 2:
            for i in range(sz[0]):
                for j in range(sz[1]):
                    flat_vec.append(value[i,j])
        else:
            for i in range(sz[0]):
                flat_vec.append(value[i])
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


def generate_iterpolation_loss(checkpoint_epochs, checkpoint_weight_vectors, model_name, args, data_loader, loss):
    n_sample = args.sample
    sample_weight_vectors = []
    print(" Start Interpolation Sampling ... ")
    for i in range(len(checkpoint_epochs)-1):
        for j in range(n_sample):
            tmp_vector = (checkpoint_weight_vectors[i] * (j+1) + checkpoint_weight_vectors[i+1] * (n_sample - j)) / (n_sample + 1)
            sample_weight_vectors.append(tmp_vector)
    sample_weight_vectors = np.array(sample_weight_vectors)
    print(" Interpolation Sampling Completed \n")
    
    all_weight_vectors = []
    for i in range(len(checkpoint_epochs)-1):
        all_weight_vectors.append(checkpoint_weight_vectors[i])
        for j in range(n_sample):
            all_weight_vectors.append(sample_weight_vectors[n_sample * i + j])
    all_weight_vectors.append(checkpoint_weight_vectors[-1])
    all_weight_vectors = np.array(all_weight_vectors)

    print(" Computing Interpolation loss ...")
    all_weight_vectors_shape = all_weight_vectors.shape
    all_loss = []
    for i in range(all_weight_vectors_shape[0]):
        test_model = eval(model_name)()
        vector_state_dict = generate_state_dict(all_weight_vectors[i], model_name)
        test_model.load_state_dict(vector_state_dict)
        
        if args.cuda: test_model.cuda()
        
        sys.stdout.write('\b'*50)
        sys.stdout.flush()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            target_dtype = str(target.dtype)
            data = torch.FloatTensor(data)
            target = torch.FloatTensor(target) if target_dtype[0] == 'f' else torch.LongTensor(target)
            data, target = Variable(data), Variable(target)

            if args.cuda: data, target = data.cuda(), target.cuda()

            output = test_model(data)
            batch_loss = loss(output, target)
            total_loss += batch_loss.data[0]

        avg_loss = total_loss / len(data_loader)
        if avg_loss > args.min_loss:
            all_loss.append(args.min_loss)
        else:
            all_loss.append(avg_loss)
        sys.stdout.write("({}) loss = {}".format(i+1, avg_loss))
        sys.stdout.flush()
    print("")
    return np.linspace(1,10000, len(all_loss)), np.array(all_loss)
    
    
    

if __name__ == '__main__':
    main()
