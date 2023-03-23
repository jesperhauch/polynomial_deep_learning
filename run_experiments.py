import shlex, subprocess

commands = []
#models = ["ccp", "polynomial_nn"]
noise = [False, True]
change_std = [False, True]

polynomial = "lambda a, b: a**10-b**9"
polynomial_name = "a10-b9"
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