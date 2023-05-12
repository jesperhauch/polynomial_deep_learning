import shlex, subprocess

commands = []
multiplication_net = ["PDCLow", "PDC", "CCP"]
epochs = 100
n_degree = 3
lag_size = 1
seq_len = 4 
loss_fn = ['MSELoss']

for net in multiplication_net:
    command = 'python training_epidemiology.py --multiplication_net {sim}'.format(sim=net)
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    command += ' --lag_size {lag}'.format(lag=lag_size)
    command += ' --seq_len {seq}'.format(seq=seq_len)
    for l in loss_fn:
        command += ' --loss_fn {loss}'.format(loss=l)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
    