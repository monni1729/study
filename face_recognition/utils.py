import torch


def accuracy(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        return correct.mul_(100.0 / batch_size)

def custom_range(length, num):
    result = []
    for i in range(num):
        temp = (i + 1) / num
        temp *= length
        result.append(int(temp))

    return result