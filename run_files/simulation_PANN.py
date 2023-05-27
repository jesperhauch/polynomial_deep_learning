import shlex, subprocess

commands = []
multiplication_net = ["PANN"]
epochs = 100
n_degree = 3   # 3, 7, 15, 31, 63
lag_size = 1    # 1, 2, 3, 4 ,5
#seq_len = 120
seq_len = [2, 6, 12, 24, 30, 40, 60, 120]

for seq in seq_len:
    command = 'python training/training_epidemiology.py --multiplication_net PANN'
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    command += ' --lag_size {lag}'.format(lag=lag_size)
    command += ' --seq_len {seq}'.format(seq=seq)
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
    