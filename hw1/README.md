# MLDS2018SPRING

## HW1

### 1-1 Deep vs. Shallow
  * **Training**
    1. Training DNN on functions
       ```
       python main.py --batch-size 16 --epochs 20000 --save-freq 100 \
              --target-func [sin, sinc, ceil, damp] --arch [deep, middle, shallow]
       ```
       ```verilog
       optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
       ```
    2. Training CNN on MNIST
       ```
       python main.py --batch-size 128 --epochs 500 --save-freq 20 \
              --dataset mnist --arch [deep, middle, shallow]
       ```
       ```python
       optimizer = optim.Adam(model.parameters())
       ```
    3. Training CNN on CIFAR-10
       ```
       python main.py --batch-size 128 --epochs 500 --save-freq 20 \
              --dataset cifar --arch [deep, middle, shallow]
       ```
       ```python
       optimizer = optim.Adam(model.parameters())
       ```
  * **Visualization**
    1. Plot loss-epoch graph
       ```
       python plot_loss_accuracy.py
       ```
    2. Plot accuracy-epoch graph (for MNIST/CIFAR-10)
       ```
       python plot_loss_accuracy.py
       ```
    3. Plot function and predicted values (for function regression)
       ```
       python plot_function.py
       ```

### 1-2 Optimization
  * **Training**
    1. Training for visualizing optimization process
       ```
       python main.py --batch-size 128 --epochs 60 --save-freq 1 \
              --dataset mnist --arch deep --save-dir models/saved/1-2-1/[1-8]
       ```
    2. Training for visualizing gradient norm

       (1) MNIST
          ```
          python main.py --batch-size 256 --epochs 10000 --save-freq 200 \
                  --dataset mnist --arch deep --save-grad
          ```
       (2) sinc
          ```
          python main.py --batch-size 128 --epochs 10000 --save-freq 200 \
                  --arch deep --save-grad
          ```
    3. What happens when gradient norm is almost zero?

       run ```vis/hw1_2/train_min_ratio.py``` first
       ```
       python train_min_ratio.py
       ```

       then run ```vis/hw1_2/plot_min_ratio.py``` for visualization
       ```
       python plot_min_ratio.py
       ```

  * **Visualization**
    1. Visualizing dimensional reduced model parameters in training process
       ```
       python plot_param_pca.py
       ```
    2. Visualizing gradient norm of model in training process
       ```
       python plot_grad_norm.py
       ```

### 1-3 Generalization
#### 1-3-1 Can network fit random labels?
##### Training
  * Run:
    ```
    python main.py --batch-size 256 --epochs 500 --save-freq 20 \
              --dataset mnist --arch deeper --rand-label --validation-split 0.1
    ```
