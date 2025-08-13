import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm

# 定义自定义进度条回调
class ProgressBarCallback(xgb.callback.TrainingCallback):
    def __init__(self, total_rounds):
        self.total_rounds = total_rounds
        self.pbar = tqdm(total=total_rounds, desc="Training Progress")

    def after_iteration(self, model, epoch, evals_log):
        if epoch % 100 == 0:
            eval_error = evals_log["eval"]["mae"][-1]
            print(f"Iteration {epoch}, Eval MAE: {eval_error}")
        self.pbar.update(1)

        if hasattr(model, 'best_iteration') and model.best_iteration == epoch:
            self.pbar.close()
            return True  # Stop training

        if epoch + 1 == self.total_rounds:
            self.pbar.close()
        return False  # Continue training


# 加载数据
# train_input_path = 'sensor_inputs.csv'
# train_output_path = 'mocap_targets.csv'
# test_input_path = 'sensor_test.csv'
# test_output_path = 'mocap_test.csv'

# train_input_path = 'teresa_sensor_unnorm.csv'
# train_output_path = 'teresa_mocap_unnorm.csv'
# test_input_path = 'teresa_sensor_test.csv'
# test_output_path = 'teresa_mocap_test.csv'

train_input_path = 'output_outliers/sensor_train.csv'
train_output_path = 'output_outliers/mocap_train.csv'
test_input_path = 'output_outliers/sensor_test.csv'
test_output_path = 'output_outliers/mocap_test.csv'

X_train = pd.read_csv(train_input_path).values
y_train_full = pd.read_csv(train_output_path).values
X_test = pd.read_csv(test_input_path).values
y_test_full = pd.read_csv(test_output_path).values

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_full.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test_full.shape}")

# 标准化输入和输出
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 对每个输出维度分别标准化
y_train_scaled = scaler_y.fit_transform(y_train_full)
y_test_scaled = scaler_y.transform(y_test_full)

# 初始化存储预测值和R2分数的列表
y_train_pred_list = []
y_test_pred_list = []
individual_train_r2_scores = []
individual_test_r2_scores = []

# 遍历每个输出维度
for i in range(y_train_full.shape[1]):
    y_train = y_train_scaled[:, i]
    y_test = y_test_scaled[:, i]

    # 转换为DMatrix格式
    train_matrix = xgb.DMatrix(X_train_scaled, label=y_train)
    test_matrix = xgb.DMatrix(X_test_scaled, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'alpha': 0.1,
        'lambda': 1,
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda'
    }

    # 训练模型
    model = xgb.train(
        params,
        train_matrix,
        num_boost_round=5000,
        evals=[(test_matrix, "eval")],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    # 收集预测值
    y_train_pred_list.append(model.predict(train_matrix))
    y_test_pred_list.append(model.predict(test_matrix))

    # 计算R²分数
    train_r2 = r2_score(y_train, y_train_pred_list[-1])
    test_r2 = r2_score(y_test, y_test_pred_list[-1])
    individual_train_r2_scores.append(train_r2)
    individual_test_r2_scores.append(test_r2)
    print(f"Output Dimension {i + 1} - Training R2 Score: {train_r2}, Testing R2 Score: {test_r2}")

# 堆叠预测值
y_train_pred_scaled = np.column_stack(y_train_pred_list)
y_test_pred_scaled = np.column_stack(y_test_pred_list)

# 计算整体R²分数
overall_train_r2 = r2_score(y_train_scaled, y_train_pred_scaled)
overall_test_r2 = r2_score(y_test_scaled, y_test_pred_scaled)

# 显示结果
print("\nIndividual Training R2 Scores per dimension:", individual_train_r2_scores)
print("Individual Testing R2 Scores per dimension:", individual_test_r2_scores)
print("Overall Training R2 Score:", overall_train_r2)
print("Overall Testing R2 Score:", overall_test_r2)