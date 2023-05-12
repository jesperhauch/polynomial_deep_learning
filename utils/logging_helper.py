from utils.model_utils import count_parameters
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
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
    if stage == "fit":
        val_r2 = r2_score(y_true, y_pred)
        val_mse = mean_squared_error(y_true, y_pred)
        logger.log_metrics({"val_r2": val_r2,
                            "val_mse": val_mse})
        print("Validation")
        print(f"R2: {val_r2:.3f}")
        print(f"MSE: {val_mse:.3f}")
        print()
    if stage == "test":
        test_r2 = r2_score(y_true, y_pred)
        test_mse = mean_squared_error(y_true, y_pred)
        test_mape = mean_absolute_percentage_error(y_true, y_pred)
        test_mae = mean_absolute_error(y_true, y_pred)
        logger.log_metrics({'test_r2': test_r2,
                            "test_mse": test_mse,
                            'test_mae': test_mae,
                            "test_mape": test_mape})
        print("Test")
        print(f"R2: {test_r2:.3f}")
        print(f"MAPE: {test_mape:.3f}")
        print(f"MAE: {test_mae:.3f}")
        print(f"MSE: {test_mse:.3f}")
        print()

def baseline_epidemiology_metrics(logger, y_true, y_pred, stage: str):
    log_features = ["S", "I", "R"]
    val_metrics = ["r2", "mape", "mse"]
    test_metrics = ["r2", "mape", "mae", "mse"]
    if stage == "fit":
        val_r2 = r2_score(y_true, y_pred, multioutput="raw_values")
        val_mape = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
        val_mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
        values = [val_r2, val_mape, val_mse]
        logger.log_metrics({f"val_{metric}_{feat}": values[i][j] for i, metric in enumerate(val_metrics) for j, feat in enumerate(log_features)})

        print("Validation")
        print(f"R2: {val_r2}")
        print(f"MAPE: {val_mape}")
        print(f"MSE: {val_mse}")
        print()

    if stage == "test":
        test_r2 = r2_score(y_true, y_pred, multioutput="raw_values")
        test_mse = mean_squared_error(y_true, y_pred, multioutput="raw_values")
        test_mape = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
        test_mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
        values = [test_r2, test_mape, test_mae, test_mse]
        logger.log_metrics({f"test_{metric}_{feat}": values[i][j] for i, metric in enumerate(test_metrics) for j, feat in enumerate(log_features)})

        print("Test")
        print(f"R2: {test_r2}")
        print(f"MAPE: {test_mape}")
        print(f"MAE: {test_mae}")
        print(f"MSE: {test_mse}")
        print()
