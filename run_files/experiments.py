import shlex, subprocess

commands = []
models = ["PDC", "PDCLow", "CCP"]
change_std = [False, True]

polynomial = "lambda a, b, c, d, e: (a**3)*(b**3)*(c**3) - d**4*e**4"
polynomial_name = "a3_b3_c3-d4_e4"
n_degree = 8
epochs = 100

for model in models:
    for std in change_std:
        command = 'python training.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
        if std:
            command += ' --data_std 1'
        command += ' --n_degree {deg}'.format(deg=n_degree)
        command += ' --epochs {epoch}'.format(epoch=epochs)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)