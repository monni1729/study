import numpy as np
import random

def calculate(original, thumb):
    result = []
    for i in range(thumb.shape[0]):
        temp = original - thumb[i]
        result.append(np.sum(temp**2))
    
    return np.array(result)


def main():
    rand_list = random.sample(list(range(11350)), 1000)
    original = np.load('original_vector.npy', allow_pickle=True)
    thumb = np.load('thumb_vector.npy', allow_pickle=True)
    
    new_original = np.zeros((1000, 2048))
    new_thumb = np.zeros((1000, 2048))
    for idx, rl in enumerate(rand_list):
        new_original[idx, :] = original[rl, : ]
        new_thumb[idx, :] = thumb[rl, : ]
    
    original = new_original
    thumb = new_thumb
    print(original.shape, thumb.shape)
    
    total = original.shape[0]
    correct = 0
    for i in range(original.shape[0]):
        a = calculate(original[i,:], thumb)
        print(i, '\t', a.argsort()[:1])
        if i in list(a.argsort()[:1]):
            correct += 1
        
        # if i == 999:
        #     break
    print(correct / total)
    print(correct)
        
if __name__ == '__main__':
    main()