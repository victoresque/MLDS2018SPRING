# MLDS2018SPRING

## HW1

### 1-1
  * **Training**
    1. Training DNN on functions
       ```
       python main.py --batch-size 16 --epochs 20000 --save-freq 100 --target-func [sin, sinc, ceil, damp] --arch [deep, middle, shallow]
       ```
       ```verilog
       optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
       ```
    2. Training CNN on MNIST
       ```
       python main.py --batch-size 128 --epochs 500 --save-freq 20 --dataset mnist --arch [deep, middle, shallow]
       ```
       ```python
       optimizer = optim.Adam(model.parameters())
       ```
    3. Training CNN on CIFAR-10
       ```
       python main.py --batch-size 128 --epochs 500 --save-freq 20 --dataset cifar --arch [deep, middle, shallow]
       ```
       ```python
       optimizer = optim.Adam(model.parameters())
       ```
  * **Visualization**
