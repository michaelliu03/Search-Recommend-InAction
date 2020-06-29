
# set the path-to-files
TRAIN_FILE = "./fm_train.csv"
TEST_FILE = "./fm_test.csv"

SUB_DIR = "./output"

NUM_SPLITS = 2
RANDOM_SEED = 2017

# types of columns of the dataset dataframe
CATEGORICAL_COLS = ['query_cqtn_1_cat', 'query_cqtn_2_cat', 'query_cqtn_3_cat', 'query_cqtn_4_cat', 'query_cqtn_5_cat']


BINARY_COLS= ['query_bm_1_bin', 'query_bm_2_bin', 'query_bm_3_bin', 'query_bm_4_bin', 'query_bm_5_bin']

# NUMERIC_COLS = [
#     # # binary
#     # "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin",
#     # "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
#     # "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_16_bin",
#     # "ps_ind_17_bin", "ps_ind_18_bin",
#     # "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
#     # "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin",
#     # numeric
#     "ps_reg_01", "ps_reg_02", "ps_reg_03",
#     "ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15",
#
#     # feature engineering
#     "missing_feat", "ps_car_13_x_ps_reg_03",
# ]

NUMERIC_COLS = ['query_cqtn_1_cat', 'query_cqtn_2_cat', 'query_cqtn_3_cat', 'query_cqtn_4_cat', 'query_cqtn_5_cat', 'query_sotf_1', 'query_sotf_2', 'query_sotf_3', 'query_sotf_4', 'query_sotf_5', 'query_motf_1', 'query_motf_2', 'query_motf_3', 'query_motf_4', 'query_motf_5', 'query_motf_1', 'query_motf_2', 'query_motf_3', 'query_motf_4', 'query_motf_5', 'query_motf_1', 'query_motf_2', 'query_motf_3', 'query_motf_4', 'query_motf_5', 'query_votf_1', 'query_votf_2', 'query_votf_3', 'query_votf_4', 'query_votf_5', 'query_other_1', 'query_other_1']


IGNORE_COLS = []


