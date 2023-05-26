from utils.model_utils import count_parameters
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from torchmetrics import MultioutputWrapper
from models.utils import RootRelativeSquaredError
import torch


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

def baseline_run_information(data_generator, model, **kwargs):
    print("=====================================================")
    print("Simulation:", type(data_generator).__name__)
    print("Model:", model)
    if kwargs.get("polynomial_features"):
        print("Feature degree:", kwargs.get("n_degree"))
    print("=====================================================")


def baseline_metrics(logger, y_true, y_pred, stage: str):
    rrse = RootRelativeSquaredError()
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rrse = rrse(torch.Tensor(y_pred), y_true).item()
    if stage == "fit":
        logger.log_metrics({"val_r2": r2,
                            "val_mse": mse,
                            "val_rrse": rrse})
        print("Validation")
        print(f"R2: {r2:.3f}")
        print(f"RRSE: {rrse:.3f}")
        print(f"MSE: {mse:.3f}")
        print()
    if stage == "test":
        test_mae = mean_absolute_error(y_true, y_pred)
        logger.log_metrics({'test_r2': r2,
                            'test_rrse': rrse,
                            "test_mse": mse,
                            'test_mae': test_mae})
        print("Test")
        print(f"R2: {r2:.3f}")
        print(f"RRSE: {rrse:.3f}")
        print(f"MAE: {test_mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print()

def baseline_epidemiology_metrics(logger, y_true, y_pred, stage: str):
    log_features = ["S", "I", "R"]
    val_metrics = ["r2", "rrse", "mae", "mse"]
    test_metrics = ["r2", "rrse", "mae", "mse"]
    rrse = MultioutputWrapper(RootRelativeSquaredError(), num_outputs=3)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rrse = rrse(torch.Tensor(y_pred), y_true)
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
    values = [r2, rrse, mae, mse]
    if stage == "fit":
        logger.log_metrics({f"val_{metric}_{feat}": values[i][j] for i, metric in enumerate(val_metrics) for j, feat in enumerate(log_features)})

        print("Validation")

    if stage == "test":
        logger.log_metrics({f"test_{metric}_{feat}": values[i][j] for i, metric in enumerate(test_metrics) for j, feat in enumerate(log_features)})

        print("Test")
    print(f"R2: {r2}")
    print(f"RRSE: {rrse}")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print()

