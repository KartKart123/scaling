from train import train
from matplotlib import pyplot as plt
from dataset import LMTextDataset

# d_model, n_heads, n_layers

# configs = [[16, 2, 4],
#            [32, 4, 4],
#            [64, 4, 4],
#            [96, 4, 4],
#            [128, 4, 4],
#            [192, 4, 4],
#            [256, 4, 4],
#            [320, 4, 4],
#            [384, 4, 4],
#            [512, 4, 4]]

# configs = [[32, 4, 2],
#            [32, 4, 4],
#            [32, 4, 8],
#            [32, 4, 12],
#            [32, 4, 16],
#            [32, 4, 20]]

configs = [[512, 8, 4]]

run = 1
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

plt.figure()
for idx, data_list in enumerate(data_lists):
    tokens = [entry['tokens'] for entry in data_list]
    loss = [entry['loss'] for entry in data_list]
    plt.plot(tokens, loss, label=f'Run {idx + 1}')

plt.xscale('log')
plt.xlabel('Tokens')
plt.ylabel('Test Loss')
plt.legend()
plt.grid(True)

plt.show()