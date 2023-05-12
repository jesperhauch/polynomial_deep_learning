# Epidemiology
R_net could have been run with n_degree//2 to reduce complexity or the amount of network parameters. It actually seems to increase performance.

## 1-step and 1-lag (start state $t_0$ and first state $t_1$)
```
import shlex, subprocess

commands = []
multiplication_net = ["PDCLow", "PDC", "CCP"]
epochs = 100
n_degree = 2
lag_size = 1
seq_len = 2
apply_scaling = [False, True]
loss_fn = ['MSELoss', 'WeightedMSELoss']
for net in multiplication_net:
    command = 'python epidemiology_training.py --multiplication_net {sim}'.format(sim=net)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    command += ' --seq_len {seq}'.format(seq=seq_len)
    for i in apply_scaling:
        if i:
            command += ' --apply_scaling'
            commands.append(command)
        else:
            for j in loss_fn:
                command += ' --loss_fn {loss}'.format(loss=j)
                commands.append(command)    

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```

## 2-step and 1-lag (start state $t_0$ and first state $t_1$)
```
import shlex, subprocess

commands = []
multiplication_net = ["PDCLow", "PDC", "CCP"]
epochs = 100
n_degree = 2
lag_size = 1
seq_len = 3
loss_fn = ['MSELoss', 'WeightedMSELoss']
for net in multiplication_net:
    command = 'python epidemiology_training.py --multiplication_net {sim}'.format(sim=net)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    command += ' --seq_len {seq}'.format(seq=seq_len)
    for j in loss_fn:
        command += ' --loss_fn {loss}'.format(loss=j)
        commands.append(command)    

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```

Examples
```
python epidemiology_training.py --multiplication_net CCP --lag_size 2 --n_degree 4 --epochs 100
python epidemiology_training.py --multiplication_net CCP --lag_size 1 --n_degree 2 --seq_len 2 --epochs 100
```

## 1-step and 2-lag
```
import shlex, subprocess

commands = []
multiplication_net = ["PDCLow", "PDC", "CCP"]
epochs = 100
n_degree = 4
lag_size = 2
seq_len = 3
apply_scaling = [False, True]
loss_fn = ['MSELoss', 'WeightedMSELoss']
for net in multiplication_net:
    command = 'python epidemiology_training.py --multiplication_net {sim}'.format(sim=net)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    command += ' --seq_len {seq}'.format(seq=seq_len)
    for i in apply_scaling:
        if i:
            command += ' --apply_scaling'
            commands.append(command)
        else:
            for j in loss_fn:
                command += ' --loss_fn {loss}'.format(loss=j)
                commands.append(command)    

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```
