import sys
sys.path.append("...")
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../..")
from models.models import DeepMnistCNN
from models.models import DeepCifarCNN


if __name__ == '__main__':
	batch_size_list = [32, 64, 128, 160, 192, 256, 512, 640, 768, 1024, 2048]
	files = ['CIFARn','CIFAR']

	checkpoint_accuracy, checkpoint_val_accuracy = [], []
	checkpoint_loss, checkpoint_val_loss = [], []

	for file in files:
		for batch_size in batch_size_list:	
			checkpoint = torch.load('../../models/saved/1-3-bonus/'+ file+'/' + str(batch_size)\
			                +'/DeepCifarCNN_cifar_checkpoint_epoch1000.pth.tar')
			epoch = checkpoint['epoch']
			logger = checkpoint['logger']
			checkpoint_loss.append(logger.entries[epoch]['loss'])
			checkpoint_accuracy.append(logger.entries[epoch]['accuracy'])
			checkpoint_val_loss.append(logger.entries[epoch]['val_loss'])
			checkpoint_val_accuracy.append(logger.entries[epoch]['val_accuracy'])
			
			
	noise_accuracy = np.array(checkpoint_accuracy[:len(batch_size_list)])
	withoutnoise_accuracy = np.array(checkpoint_accuracy[len(batch_size_list):])

	val_noise_accuracy = np.array(checkpoint_val_accuracy[:len(batch_size_list)])
	val_withoutnoise_accuracy = np.array(checkpoint_val_accuracy[len(batch_size_list):])

	
	noise_loss = np.array(checkpoint_loss[:len(batch_size_list)])
	withoutnoise_loss = np.array(checkpoint_loss[len(batch_size_list):])

	val_noise_loss = np.array(checkpoint_val_loss[:len(batch_size_list)])
	val_withoutnoise_loss = np.array(checkpoint_val_loss[len(batch_size_list):])


	batch_size_list = np.array(batch_size_list)

	# sensitivity
	#sensitivity = (noise_accuracy[1:] - withoutnoise_accuracy[:5] /  \
	#					(noise_accuracy[1:]))
	#print (sensitivity)
	fig = plt.figure(figsize=(12, 6))

	ax1 = fig.add_subplot(1,2,1)
	ln1 = ax1.semilogx(batch_size_list, noise_loss, 'b-', label='loss with noise')
	ln2 = ax1.semilogx(batch_size_list, withoutnoise_loss, 'b--', label='loss without noise')
	#ln3 = ax1.semilogx(batch_size_list, val_noise_loss, 'r-', label='validation loss with noise')
	#ln4 = ax1.semilogx(batch_size_list, val_withoutnoise_loss, 'r--', label='validation loss without noise')
	ax1.set_xlabel('batch size')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax1.set_ylabel('loss', color='b')
	ax1.tick_params('y', colors='b')

	lns1 = ln1+ln2#+ln3+ln4
	labs1 = [l.get_label() for l in lns1]
	ax1.legend(lns1, labs1, loc=0)
	ax1.set_title('CIFAR-10 noise vs. loss')


	ax2 = fig.add_subplot(1,2,2)
	ln5 = ax2.semilogx(batch_size_list, noise_accuracy, 'r-', label='accuracy with noise')
	ln6 = ax2.semilogx(batch_size_list, withoutnoise_accuracy, 'r--', label='accuracy without noise')
	#ln7 = ax2.semilogx(batch_size_list, val_noise_accuracy, 'r-', label='validation accuracy with noise')
	#ln8 = ax2.semilogx(batch_size_list, val_withoutnoise_accuracy, 'r--', label='validation accuracy without noise')
	#ax3 = ax2.twinx()
	#ln9 = ax3.semilogx(batch_size_list[1:], sensitivity, 'b--', label='sensitivity')

	ax2.set_xlabel('batch size')
	# Make the y-axis label, ticks and tick labels match the line color.
	ax2.set_ylabel('accuracy', color='r')
	ax2.tick_params('y', colors='r')

	lns2 = ln5+ln6#+ln7+ln8
	labs2 = [l.get_label() for l in lns2]
	ax2.legend(lns2, labs2, loc=0)
	ax2.set_title('CIFAR-10 noise vs. accuracy')

	fig.tight_layout()
	plt.show()
    #savefig_path = './{}_sensitivity.png'.format(args.dataset)
	#fig.savefig("CIFAR.png")

