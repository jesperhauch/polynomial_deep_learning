import shlex, subprocess

commands = []
models = ["PANN", "PDC", "PDCLow", "CCP"]
polynomial_dict = {"2a3_6b2": ["lambda a, b: 2*a**3 * 6*b**2", 5],
                   "2a3_b2-3c": ["lambda a, b, c: 2*a**3 * b**2 - 3*c", 5],
                   "2a3_b3-3c": ["lambda a, b, c: 2*a**3 * b**3 - 3*c", 6],
                   "2a2_b2_c2-3d": ["lambda a, b, c, d: 2*a**2 * b**2 * c**2 - 3*d", 6],
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
        command = 'python training/training_polynomials.py --model {m} --polynomial "{poly}" --polynomial_name {poly_name}'.format(m=model, poly=polynomial, poly_name=polynomial_name)
        command += ' --n_degree {deg}'.format(deg=n_degree)
        command += ' --epochs {epoch}'.format(epoch=epochs)
        commands.append(command)

if __name__ == "__main__":
    for command in commands:
        args = shlex.split(command)
        subprocess.run(args)