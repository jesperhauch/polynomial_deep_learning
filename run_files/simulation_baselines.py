import shlex, subprocess

commands = []
#models = ["LinearRegression", "GradientBoostingRegressor"]
models = ["RandomForestRegressor"]
n_degree = 15   # 3, 7, 15, 31, 63
lag_size = 3    # 1, 2, 3, 4 ,5
seq_len = 120

for i, model in enumerate(models):
    command = 'python training/baseline_epidemiology_training.py --model {m}'.format(m=model)
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --lag_size {lag}'.format(lag=lag_size)
    command += ' --seq_len {seq}'.format(seq=seq_len)
    command += ' --n_data 10000'
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)