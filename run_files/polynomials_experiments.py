import shlex, subprocess

commands = []
models = ["PANN", "PDC", "PDCLow", "CCP"]

polynomial = "lambda a, b: 5*a**2 + 6*b**2"
polynomial_name = "5a2+6b2"
n_degree = 2
epochs = 30

for model in models:
    command = 'python training_polynomials.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)