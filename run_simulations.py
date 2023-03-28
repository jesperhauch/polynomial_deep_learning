import shlex, subprocess

commands = []
models = ["pdc", "pdclow", "ccp"]

simulation = 'ShortColumn'
n_degree = 7
epochs = 50

for model in models:
    command = 'python training.py --model {m} --data_generator {sim}'.format(m=model, sim=simulation)
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)