from train import train
from matplotlib import pyplot as plt

# d_model, n_heads, n_layers

# configs = [[32, 4, 4],
#           [64, 4, 4],
#           [96, 4, 4],
#           [128, 4, 4]]
# configs = [[96, 4, 8],
#            [128, 4, 8],
#            [160, 4, 8],
#            [256, 4, 4]]

configs = [[16, 2, 4],
           [32, 4, 4],
           [64, 4, 4],
           [96, 4, 4],
           [128, 4, 4],
           [192, 4, 4],
           [256, 4, 4],
           [320, 4, 4],
           [384, 4, 4],
           [512, 4, 4]]

run = 0
data_lists = []
for config in configs:
    run += 1
    d_model, n_heads, n_layers = config
    data_lists.append(train(d_model, n_heads, n_layers, run))

plt.figure()
for idx, data_list in enumerate(data_lists):
    compute = [entry['compute'] for entry in data_list]
    loss = [entry['loss'] for entry in data_list]
    plt.plot(compute, loss, label=f'Run {idx + 1}')

plt.xscale('log')
plt.xlabel('Compute (FLOPs)')
plt.ylabel('Test Loss')
plt.legend()
plt.grid(True)
plt.show()