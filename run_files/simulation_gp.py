import shlex, subprocess

commands = []
model = "GaussianProcessRegressor"
seq_len = 120
degrees = [3, 7, 15, 31, 63]
lag_sizes = [1, 2, 3, 4, 5]
#n_degree = 3   # 3, 7, 15, 31, 63
#lag_size = 1    # 1, 2, 3, 4 ,5

for i, n_degree in enumerate(degrees):
    command = 'python training/baseline_epidemiology_training.py --model {m}'.format(m=model)
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --lag_size {lag}'.format(lag=lag_sizes[i])
    command += ' --seq_len {seq}'.format(seq=seq_len)
    command += ' --n_data 100'
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)