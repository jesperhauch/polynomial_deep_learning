import shlex, subprocess

commands = []
multiplication_net = ["PDCLow", "PDC", "CCP"]
epochs = 100
n_degree = 63   # 3, 7, 15, 31, 63
lag_size = 5    # 1, 2, 3, 4 ,5
seq_len = 120

for net in multiplication_net:
    command = 'python training/training_epidemiology.py --multiplication_net {sim}'.format(sim=net)
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    command += ' --lag_size {lag}'.format(lag=lag_size)
    command += ' --seq_len {seq}'.format(seq=seq_len)
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
    