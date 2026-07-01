# config.py

OrigData_path = '../CMAQ_Oringin_Data'
Train_Data_path = './CMAQ_Train_Data'
Trained_Model_path = './CMAQ_Model'

HIST_DIR = './HIST_DIR_o3'   # PM2.5_obs 歷史數據資料夾
year_of_data = 2016       # 定義目前數據資料年分，以此判定平閏年，設定暫定 time 的數據長度

#####################################################################
#         You can change variables that under this hint             #
#                        Arguments Cross test                       #
#####################################################################

# Please set only one site.
site_sets = [42]

# Input_timestep (past)
Input_timesteps = 72

# Output_timestep (future N)
Output_timesteps = 72

# -------------------------------
# Dual-Branch Feature Setting
# -------------------------------
# Past branch: t-Input_timesteps ~ t-1
#X_PAST_COLS = ['PM2.5_obs']
X_PAST_COLS = ['O3_obs']

# Future branch: t ~ t+Output_timesteps-1
X_FUTURE_COLS = [
    'PM2.5', 'O3', 'NMHC', 'RGRND', 'NO2', 'NOx',
    'TEMP2_JD', 'WSPD10', 'WDIR10',
    'PRSFC', 'PBL', 'QV', 'CFRAC'
]

# Branch input dimensions
PAST_INPUT_DIM = len(X_PAST_COLS)
FUTURE_INPUT_DIM = len(X_FUTURE_COLS)

# ================================
# Model Hyperparameters
# ================================

BATCH_SIZE = 16

TCN_CHANNELS = [128, 256, 512]
KERNEL_SIZE = 2

TRANSFORMER_D_MODEL = 512
TRANSFORMER_NHEAD = 16
TRANSFORMER_DIM_FEEDFORWARD = 128
TRANSFORMER_DROPOUT = 0.2

# ================================
# 🔥 Fusion MLP (新增區塊)
# ================================

# concat 後維度 = 512(past) + 512(future) = 1024
FUSION_INPUT_DIM = TCN_CHANNELS[-1] * 2  # = 1024

# MLP hidden layer
FUSION_HIDDEN_DIM = 512

# Activation function for MLP fusion
# 可選: "ReLU", "GELU", "LeakyReLU", "Tanh"
FUSION_PRELU_NUM_PARAMETERS = 1
FUSION_ACTIVATION = "PReLU"

# ================================
# Target column
# ================================

#Y_COL = 'PM2.5_diff'
Y_COL = 'O3_diff'
Need_element_Y = [Y_COL]


# 用來從 txt 撈資料的 cls 清單（至少要包含 past + future 欄位）
Need_element_X = sorted(list(set(X_PAST_COLS + X_FUTURE_COLS)))

# 輸出維度仍由 Output_timesteps 決定
OUTPUT_DIM = Output_timesteps

# -------------------------------
# For Log Name (keep)
# -------------------------------
LossFunction = "RMSE"
ActivateFunction = "PRelu"

# Train / Valid
Train_ratio = 8
Valid_ratio = 2

# Train Month / Predict Month
Train_month = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
predict_month = [1]

#####################################################################
#               Other variable Save Name                            #
#####################################################################

_fileName_sets = ["201601d4_sim.met", "201601d4_sim.met2", "201601d4_sim_obs"]

# In dataset.py
train_data_name = f'{Train_Data_path}/{site_sets[0]}_training_data.pt'

LogName = (
    f'{site_sets[0]}_{Input_timesteps}_{Output_timesteps}_'
    f'{LossFunction}_{ActivateFunction}_{Train_ratio}_{Valid_ratio}_'
    f'predict{predict_month[0]}_{Need_element_Y[0]}'
)

best_model_path = f'{Trained_Model_path}/{LogName}_best_model.pth'
pre_tru = f'{Trained_Model_path}/{LogName}_pre_tru_{Need_element_Y[0]}'
