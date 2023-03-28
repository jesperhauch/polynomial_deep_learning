from utils.model_utils import count_parameters
def run_information(logger, data_generator, model, **kwargs):
    logger.log_hyperparams(kwargs)
    if type(data_generator).__name__ == "NormalGenerator":
        print("=====================================================")
        print("Polynomial:", kwargs.get("polynomial_name"))
        print("Model:", model)
        print("# model parameters:", count_parameters(model))
        print("Train distribution:", kwargs.get("data_dist"))
        print("Non-deterministic:", kwargs.get("noise"))
        print("=====================================================")
    else:
        ("=====================================================")
        print("Simulation:", type(data_generator).__name__)
        print("Model:", model)
        print("# model parameters:", count_parameters(model))
        print("Non-deterministic:", kwargs.get("noise"))
        print("=====================================================")