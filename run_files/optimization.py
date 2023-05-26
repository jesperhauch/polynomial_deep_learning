import shlex, subprocess

commands = []
models = ["PANN", "PDCLow", "CCP", "PDC"]
simulations = {"Currin": 2, "Bukin06": 2, "Price03": 4,
               "DettePepelyshev": 4, "Colville": 4, "LimPolynomial": 5,
               "CamelThreeHump": 6, "Beale": 8, "GoldsteinPrice": 8 
               }
epochs = 100

for sim, n_degrees in simulations.items():
    for model in models:
        command = 'python training/training_polynomials.py --model {m} --data_generator {sim}'.format(m=model, sim=sim)
        command += ' --n_degree {deg}'.format(deg=n_degrees)
        command += ' --epochs {epoch}'.format(epoch=epochs)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)