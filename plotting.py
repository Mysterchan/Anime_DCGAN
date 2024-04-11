import matplotlib.pyplot as plt

def plot(images):
    figure, axes = plt.subplots(1,4,figsize=(64,64))
    i = 0
    for ax in axes:
        ax.imshow(images[i])
        ax.axis("off")
        i+=1
    plt.show()
