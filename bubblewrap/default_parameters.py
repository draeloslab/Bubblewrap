# ## Parameters
# N = 1000             # number of nodes to tile with
# lam = 1e-3          # lambda
# nu = 1e-3           # nu
# eps = 1e-3          # epsilon sets data forgetting
# step = 8e-2         # for adam gradients
# M = 30              # small set of data seen for initialization
# B_thresh = -10      # threshold for when to teleport (log scale)
# batch = False       # run in batch mode
# batch_size = 1      # batch mode size; if not batch is 1
# go_fast = False     # flag to skip computing priors, predictions, and entropy for optimal speed

default_rwd_parameters = dict(
    num=200,
    lam=1e-3,
    nu=1e-3,
    eps=1e-3,
    step=8e-2,
    M=30,
    B_thresh=-10,
    batch=False,
    batch_size=1,
    go_fast=False,
    lookahead_steps=[1, 2, 5, 10],
    seed=42,
    save_A=False,
    balance=1,
    beh_reg_constant_term=True
)

default_clock_parameters = dict(
    num=8,
    lam=1e-3,
    nu=1e-3,
    eps=1e-4,
    step=8e-2,
    M=100,
    B_thresh=-5,
    batch=False,
    batch_size=1,
    go_fast=False,
    lookahead_steps=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50],
    seed=42,
    save_A=False,
    balance=1,
    beh_reg_constant_term=True
)

# reasonable_parameter_ranges = dict(
#     ?
# )
