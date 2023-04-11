import shlex, subprocess

commands = []
models = ["LinearRegression", "GradientBoostingRegressor", "RandomForestRegressor"]
simulations = {"Currin": 2, "Bukin06": 2, "Price03": 4,
               "DettePepelyshev": 4, "Colville": 4, "LimPolynomial": 5,
               "CamelThreeHump": 6, "Beale": 8, "GoldsteinPrice": 8 
               }
add_polynomial_features = [False, True]

for sim, n_degrees in simulations.items():
    for model in models:
        for poly in add_polynomial_features:
            command = 'python baseline_training.py --model {m} --data_generator {sim}'.format(m=model, sim=sim)
            command += ' --n_degree {deg}'.format(deg=n_degrees)
            if poly:
                command += ' --polynomial_features'
            commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)