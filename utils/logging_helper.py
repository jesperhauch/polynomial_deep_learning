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
        val_rmse = mean_squared_error(y_true, y_pred, squared=False)
        logger.log_metrics({"val_r2": val_r2,
                            "val_rmse": val_rmse})
        print("Validation")
        print(f"R2: {val_r2:.3f}")
        print(f"RMSE: {val_rmse:.3f}")
        print()
    if stage == "test":
        test_r2 = r2_score(y_true, y_pred)
        test_rmse = mean_squared_error(y_true, y_pred, squared=False)
        test_mape = mean_absolute_percentage_error(y_true, y_pred)
        test_mae = mean_absolute_error(y_true, y_pred)
        logger.log_metrics({'test_r2': test_r2,
                            "test_rmse": test_rmse,
                            'test_mae': test_mae,
                            "test_mape": test_mape})
        print("Test")
        print(f"R2: {test_r2:.3f}")
        print(f"RMSE: {test_rmse:.3f}")
        print(f"MAE: {test_mae:.3f}")
        print(f"MAPE: {test_mape:.3f}")
        print()
