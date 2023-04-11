# Optimization Functions
```
import shlex, subprocess

commands = []
models = ["FeedForwardNN", "PolynomialNN", "PDCLow", "CCP", "PDC"]
simulations = {"Currin": 2, "Bukin06": 2, "Price03": 4,
               "DettePepelyshev": 4, "Colville": 4, "LimPolynomial": 5,
               "CamelThreeHump": 6, "Beale": 8, "GoldsteinPrice": 8 
               }
epochs = 100

for sim, n_degrees in simulations.items():
    for model in models:
        command = 'python training.py --model {m} --data_generator {sim}'.format(m=model, sim=sim)
        command += ' --n_degree {deg}'.format(deg=n_degrees)
        command += ' --epochs {epoch}'.format(epoch=epochs)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)
```

Models cannot learn at all for Price 3 Function. Models are able to learn Bukin 6 Function but can barely generalize. Models have high R2 but also high errors for GoldsteinPrice. Otherwise models are able to generalize.
$$
f_{\text{Price3}}(x_1, x_2) = 100(x_2-x_1^2)^2+6[6.4(x_2-0.5)^2-x_1-0.6]^2\\
f_{\text{Bukin6}}(x_1, x_2) = 100\sqrt{||x_2-0.01x_1^2||}+0.01||x_1+10||\\
f_{\text{Goldstein-Price}}(x_1, x_2) = \left(1 + \left(x_1 + x_2 + 1\right)^2 \cdot \left(19 - 14x_1 + 3x_1^2 - 14x_2 + 6x_1x_2 + 3x_2^2\right)\right)\left(30 + \left(2x_1 - 3x_2\right)^2 \left(18 - 32x_1 + 12x_1^22 + 48x_2 - 36x_1x_2 + 27x_2^2\right)\right)$$