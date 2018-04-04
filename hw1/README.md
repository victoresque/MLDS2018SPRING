# HW1

## Requirements
  * Python 3.6
  * PyTorch 0.3
  * torchvision
  * scikit-learn 0.19.1
  * matplotlib
  * numpy
  * MulticoreTSNE
---
## Table of Content

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [HW1](#hw1)
	* [Requirements](#requirements)
	* [Table of Content](#table-of-content)
	* [1-1 Deep vs. Shallow](#1-1-deep-vs-shallow)
		* [1-1-1 Simulate a function](#1-1-1-simulate-a-function)
		* [1-1-2 Train on actual tasks](#1-1-2-train-on-actual-tasks)
	* [1-2 Optimization](#1-2-optimization)
		* [1-2-1 Visualize the optimization process](#1-2-1-visualize-the-optimization-process)
		* [1-2-2 Observe gradient norm during training](#1-2-2-observe-gradient-norm-during-training)
		* [1-2-3 What happens when gradient is almost zero](#1-2-3-what-happens-when-gradient-is-almost-zero)
		* [1-2-B Bonus: Error surface](#1-2-b-bonus-error-surface)
	* [1-3 Generalization](#1-3-generalization)
		* [1-3-1 Can network fit random labels?](#1-3-1-can-network-fit-random-labels)
		* [1-3-2 Number of parameters vs. Generalization](#1-3-2-number-of-parameters-vs-generalization)
		* [1-3-3 Flatness vs. Generalization](#1-3-3-flatness-vs-generalization)
			* [1-3-3-1 Visualize the lines between two training approaches](#1-3-3-1-visualize-the-lines-between-two-training-approaches)
			* [1-3-3-2 Visualize the sensitivities of different training approaches](#1-3-3-2-visualize-the-sensitivities-of-different-training-approaches)
			* [1-3-3-B Bonus: Evaluate a model's ability to generalize](#1-3-3-b-bonus-evaluate-a-models-ability-to-generalize)

<!-- /code_chunk_output -->

---
## 1-1 Deep vs. Shallow

### 1-1-1 Simulate a function
  * **Training**
    **Choose one argument in [...] list**
    ```
    python main.py --batch-size 128 --epochs 20000 --save-freq 1000 \
        --target-func [damp, square] --arch [deep, middle, shallow] \
        --save-dir models/saved/1-1-1
    ```
  * **Visualization**
    **You may need to edit checkpoint paths in these scripts**
    Plot loss-epoch:
    ```
    cd vis/hw1_1; python plot_loss_accuracy.py
    ```
    Plot function prediction and ground truth:
    ```
    cd vis/hw1_1; python plot_function.py
    ```
### 1-1-2 Train on actual tasks
  * **Training**
    Train on MNIST
    ```
    python main.py --batch-size 128 --epochs 1000 --save-freq 100 \
        --dataset mnist --arch [deep, middle, shallow] \
        --save-dir models/saved/1-1-2
    ```
    Train on CIFAR-10
    ```
    python main.py --batch-size 128 --epochs 2000 --save-freq 200 \
        --dataset cifar --arch [deep, middle, shallow] \
        --save-dir models/saved/1-1-2
    ```
  * **Visualization**
    **You may need to edit checkpoint paths in these scripts**
    Plot loss-epoch and accuracy-epoch:
    ```
    cd vis/hw1_1; python plot_loss_accuracy.py
    ```

## 1-2 Optimization

### 1-2-1 Visualize the optimization process
  * **Training**
    Training on MNIST
    ```
    python main.py --batch-size 128 --epochs 99 --save-freq 3 \
        --dataset mnist --arch deep --save-dir models/saved/1-2-1
    ```
  * **Visualization**
    **You may need to edit checkpoint paths in these scripts**
    ```
    cd vis/hw1_2; python plot_param_pca.py
    ```

### 1-2-2 Observe gradient norm during training
  * **Training**
    ```
    python main.py --batch-size 128 --epochs 20000 --save-freq 2000 \
        --target-func stair --arch [deep, middle, shallow] --save-grad \
        --save-dir models/saved/1-2-2
    ```
  * **Visualization**
    **You may need to edit checkpoint paths in these scripts**
    ```
    cd vis/hw1_2; python plot_grad_norm.py
    ```

### 1-2-3 What happens when gradient is almost zero
  * **Training**
    ```
    cd vis/hw1_2; python train_min_ratio.py
    ```
  * **Visualization**
    ```
    cd vis/hw1_2; python plot_min_ratio.py
    ```

### 1-2-B Bonus: Error surface
  * **Training & Visualization**
    ```
    cd vis/hw1_2; python plot_error_surface.py
    ```

## 1-3 Generalization

### 1-3-1 Can network fit random labels?
  * **Training**
    ```
    python main.py --batch-size 128 --epochs 500 --save-freq 500 \
        --dataset mnist --arch deeper64 --rand-label --validation-split 0.1 \
        --save-dir models/saved/1-3-1
    ```
  * **Visualization**
    **You may need to edit checkpoint paths in these scripts**
    ```
    cd vis/hw1_3; python plot_random_label.py
    ```

### 1-3-2 Number of parameters vs. Generalization
  * **Training**
    ```
    cd vis/hw1_3; python train_different_parameters.py
    ```
  * **Visualization**
    ```
    cd vis/hw1_3; python plot_different_parameters.py
    ```

### 1-3-3 Flatness vs. Generalization

#### 1-3-3-1 Visualize the lines between two training approaches
  * **Training & Visualization**
    ```
    cd vis/hw1_3; python plot_interpolation.py
    ```

#### 1-3-3-2 Visualize the sensitivities of different training approaches
  * **Training & Visualization**
    ```
    cd vis/hw1_3; python plot_sensitivity.py
    ```

#### 1-3-3-B Bonus: Evaluate a model's ability to generalize
