# Running experiment
## 5a2+6b2

```
commands = []
models = ["polynomial_nn", "ccp"]
noise = [False, False, True, True]
set_std = [False, True, False, True]

polynomial = "lambda a, b: 5*a**2+6*b**2"
polynomial_name = "5a2+6b2"

for model in models:
    for i in range(len(noise)):
        command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
        if noise[i]:
            command += '_noise --noise'
        if set_std[i]:
            command += ' --data_std 1'
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```

## 2a3*6b2
```
import shlex, subprocess

commands = []
models = ["polynomial_nn", "ccp"]
noise = [False, False, True, True]
set_std = [False, True, False, True]

polynomial = "lambda a, b: 2*a**3*6*b**2"
polynomial_name = "2a3_times_6b2"
n_degree = 5

for model in models:
    for i in range(len(noise)):
        command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
        if noise[i]:
            command += ' --noise'
        if set_std[i]:
            command += ' --data_std 1'
        command += ' --n_degree {deg}'.format(deg=n_degree)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```
**Additional tests**
```
python training.py --model ccp --polynomial "lambda a, b: 2*a**3*6*b**2" --polynomial_name 2a3_times_6b2 --n_degree 5 --relu --epochs 50
python training.py --model polynomial_nn --polynomial "lambda a, b: 2*a**3*6*b**2" --polynomial_name 2a3_times_6b2 --n_degree 5 --n_hidden_layers 3 --epochs 30
python training.py --model polynomial_nn_out --polynomial "lambda a, b: 2*a**3*6*b**2" --polynomial_name 2a3_times_6b2 --n_degree 5 --epochs 30
python training.py --model polynomial_nn_each --polynomial "lambda a, b: 2*a**3*6*b**2" --polynomial_name 2a3_times_6b2 --n_degree 2 --n_hidden_layers 3 --epochs 30
```

## 2a3*b2-3c
```
import shlex, subprocess

commands = []
models = ["polynomial_nn", "ccp"]
noise = [False, False, True, True]
set_std = [False, True, False, True]

polynomial = "lambda a, b, c: (2*a**3)*b**2 - 3*c"
polynomial_name = "2a3_times_b2-3c"
n_degree = 5

for model in models:
    for i in range(len(noise)):
        command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
        if noise[i]:
            command += ' --noise'
        if set_std[i]:
            command += ' --data_std 1'
        command += ' --n_degree {deg}'.format(deg=n_degree)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```

## 2a3*b3-3c
```
import shlex, subprocess

commands = []
models = ["ccp", "polynomial_nn"]
noise = [False, True]
change_std = [False, True]

polynomial = "lambda a, b, c: 2*(a**3)*(b**3)-3*c"
polynomial_name = "2a3_b3-3c"
n_degree = 6
epochs = 30

for model in models:
    for n in noise:
        for std in change_std:
            command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
            if n:
                command += ' --noise'
            if std:
                command += ' --data_std 1'
            command += ' --n_degree {deg}'.format(deg=n_degree)
            command += ' --epochs {epoch}'.format(epoch=epochs)
            commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```
# 2a2\*b2\*c2-3c
```
import shlex, subprocess

commands = []
models = ["ccp", "polynomial_nn"]
noise = [False, True]
change_std = [False, True]

polynomial = "lambda a, b, c, d: 2*(a**2)*(b**2)*c**2 - 3*c"
polynomial_name = "2a2_b2_c2-3c"
n_degree = 10
epochs = 50

for model in models:
    for n in noise:
        for std in change_std:
            command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
            if n:
                command += ' --noise'
            if std:
                command += ' --data_std 1'
            command += ' --n_degree {deg}'.format(deg=n_degree)
            command += ' --epochs {epoch}'.format(epoch=epochs)
            commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```


## a\*b\*c\*d\*e\*f\*g
```
import shlex, subprocess

commands = []
models = ["ccp", "polynomial_nn"]
noise = [False, False, True, True]
set_std = [False, True, False, True]

polynomial = "lambda a, b, c, d, e, f, g: a*b*c*d*e*f*g"
polynomial_name = "a_b_c_d_e_f_g"
n_degree = 7
epochs = 30

for model in models:
    for i in range(len(noise)):
        command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
        if noise[i]:
            command += ' --noise'
        if set_std[i]:
            command += ' --data_std 1'
        command += ' --n_degree {deg}'.format(deg=n_degree)
        command += ' --epochs {epoch}'.format(epoch=epochs)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```

## a\*b\*c + d - e - f - g
```
import shlex, subprocess

commands = []
models = ["ccp", "polynomial_nn"]
noise = [False, True]
change_std = [False, True]

polynomial = "lambda a, b, c, d, e, f, g: a*b*c+d-e-f-g"
polynomial_name = "a_b_c+d-e-f-g"
n_degree = 3
epochs = 30

for model in models:
    for n in noise:
        for std in change_std:
            command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
            if n:
                command += ' --noise'
            if std:
                command += ' --data_std 1'
            command += ' --n_degree {deg}'.format(deg=n_degree)
            command += ' --epochs {epoch}'.format(epoch=epochs)
            commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```

# a10-b9
```
import shlex, subprocess

commands = []
models = ["ccp", "polynomial_nn"]
noise = [False, True]
change_std = [False, True]

polynomial = "lambda a, b: a**10-b**9"
polynomial_name = "a10-b9"
n_degree = 10
epochs = 30

for model in models:
    for n in noise:
        for std in change_std:
            command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
            if n:
                command += ' --noise'
            if std:
                command += ' --data_std 1'
            command += ' --n_degree {deg}'.format(deg=n_degree)
            command += ' --epochs {epoch}'.format(epoch=epochs)
            commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```

# Tensorboard
```
tensorboard --logdir=tb_logs/
```

# Environment.yml file
```
conda env export > environment.yml
```