

from src.metrics import mae,mse,r2_score

def evaluation_model(y_true,y_pred) : 
    return {
        "MSE": mse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }