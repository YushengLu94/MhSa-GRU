import random
def shuffle(dataset):
    index = [i for i in range(len(dataset))]
    random.shuffle(index)
    dataset = dataset.iloc[index]
    return  dataset