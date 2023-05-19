import shlex, subprocess

commands = []
models = ["LinearRegression", "GradientBoostingRegressor", "RandomForestRegressor"]
n_degree = 3
lag_size = 1
seq_len = 25
#add_polynomial_features = [False, True]

for i, model in enumerate(models):
    command = 'python baseline_epidemiology_training.py --model {m}'.format(m=model)
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --lag_size {lag}'.format(lag=lag_size)
    command += ' --seq_len {seq}'.format(seq=seq_len)
    #if i == 1:
    #    command += ' --poly_features'
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)