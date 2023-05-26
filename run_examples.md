# Learning Polynomials

## Arbitrary Polynomials
Running all experiments in parallel.
```
import shlex, subprocess

commands = []
models = ["PANN", "PDC", "PDCLow", "CCP"]
polynomial_dict = {"a": ["lambda a: a", 1],
                   "a2": ["lambda a: a**2", 2],
                   "5a2+6b2": ["lambda a, b: 5*a**2 + 6*b**2", 2],
                   "2a3_6b2": ["lambda a, b: 2*a**3 * 6*b**2", 5],
                   "2a3_b2-3c": ["lambda a, b, c: 2*a**3 * b**2 - 3*c", 5],
                   "2a3_b3-3c": ["lambda a, b, c: 2*a**3 * b**3 - 3*c", 6],
                   "2a3_b2_c2-3d": ["lambda a, b, c, d: 2*a**2 * b**2 * c**2 - 3*d", 6],
                   "2a3_b2_c2-d6": ["lambda a, b, c, d: 2*a**3 * b**2 * c**2 - d**6", 7],
                   "a3_b2_c3-d3_e3": ["lambda a, b, c, d, e: a**3 * b**2 * c**3 - d**3 * e**3", 8],
                   "a3_b3_c3-d4_e4": ["lambda a, b, c, d, e: a**3 * b**3 * c**3 - d**4 * e**4", 9],
                   "a3_b2_c2_d3-e5_f5": ["lambda a, b, c, d, e, f: a**3 * b**2 * c**2 * d**3 - e**5 * f**5", 10],
                   "a_b_c_d_e_f_g": ["lambda a, b, c, d, e, f, g: a*b*c*d*e*f*g", 7],
                   "a_b_c+d-e-f-g": ["lambda a, b, c, d, e, f, g: a*b*c + d - e - f - g", 3],
                   "a10-b9": ["lambda a, b: a**10 - b**9", 10]
                   }
epochs = 30

for polynomial_name, (polynomial, n_degree) in polynomial_dict.items():
    for model in models:
        command = 'python training_polynomials.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
        command += ' --n_degree {deg}'.format(deg=n_degree)
        command += ' --epochs {epoch}'.format(epoch=epochs)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
```

Run single experiments in terminal
```
python training_polynomials.py --model PANN --polynomial "lambda a: a" --polynomial_name a --n_degree 1 --epochs 30
```

## Optimization Functions
Running all optimization functions for all neural network models.
```
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
        command = 'python training.py --model {m} --data_generator {sim}'.format(m=model, sim=sim)
        command += ' --n_degree {deg}'.format(deg=n_degrees)
        command += ' --epochs {epoch}'.format(epoch=epochs)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
```

# Simulation metamodeling
Run all neural networks for 100 epochs for a fixed lag size and sequence length. `n_degree` should be updated according to the polynomial degree of the KM equations determined by the value of `lag_size`.
```
import shlex, subprocess

commands = []
multiplication_net = ["PANN", "PDCLow", "PDC", "CCP"]
epochs = 100
n_degree = 3
lag_size = 1
seq_len = 120 
loss_fn = ['MSELoss']

for net in multiplication_net:
    command = 'python training_epidemiology.py --multiplication_net {sim}'.format(sim=net)
    command += ' --n_degree {deg}'.format(deg=n_degree)
    command += ' --epochs {epoch}'.format(epoch=epochs)
    command += ' --lag_size {lag}'.format(lag=lag_size)
    command += ' --seq_len {seq}'.format(seq=seq_len)
    for l in loss_fn:
        command += ' --loss_fn {loss}'.format(loss=l)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.Popen(args)
```