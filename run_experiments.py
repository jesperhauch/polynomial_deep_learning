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