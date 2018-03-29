import json
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('min_ratio.json') as f:
        history = json.load(f)
    x = history['min_ratio']
    y = history['loss']
    plt.plot(x, y, 'o')
    plt.xlabel('minimum ratio')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.show()
