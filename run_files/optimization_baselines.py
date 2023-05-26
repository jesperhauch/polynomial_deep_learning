import shlex, subprocess

commands = []
models = ["LinearRegression", "GradientBoostingRegressor", "RandomForestRegressor"]
simulations = {"Currin": 2, "Bukin06": 2, "Price03": 4,
               "DettePepelyshev": 4, "Colville": 4, "LimPolynomial": 5,
               "CamelThreeHump": 6, "Beale": 8, "GoldsteinPrice": 8 
               }

for sim, n_degrees in simulations.items():
    for model in models:
        command = 'python training/baseline_polynomial_training.py --model {m} --data_generator {sim}'.format(m=model, sim=sim)
        command += ' --n_degree {deg}'.format(deg=n_degrees)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)