
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
import optuna
# 
optuna.logging.set_verbosity(optuna.logging.WARNING)
import tqdm
import time

# Perturbation generation
def generate_perturbation_in_advance(w_t, W):
    # 2normはsqrt(dim)Wでboundされる
    rand = np.random.normal(0, W/4, 2)
    return np.clip(w_t + rand, -W, W)

def generate_packetloss_in_advance(T, f_G, rho, H):
    """
    Generate packet loss data.
    T: Total time steps
    f_G: Global maximum frequency
    f_L: Local maximum frequency
    H: Segment length

    Returns:
        packet_loss: Array of packet loss indicators (1 for no loss, 0 for loss)
    """
    packet_loss = []
    sequence_num = int(T / H)

    for i in tqdm.tqdm(range(sequence_num)):
        if np.random.rand() < f_G:
            loss_length = max(min(int(rho), np.random.poisson(rho * 2 / 3)), 1)
            start = np.random.randint(0, H - loss_length + 1)
            packet_loss += [1] * start + [0] * loss_length + [1] * (H - start - loss_length)
        else:
            packet_loss += [1] * H

    # 配列長が T に満たない場合に調整
    packet_loss = np.array(packet_loss)
    if len(packet_loss) < T:
        repeat_num = T // len(packet_loss)
        remain_num = T % len(packet_loss)
        packet_loss = np.concatenate([np.tile(packet_loss, repeat_num), packet_loss[:remain_num]], axis=0)
    else:
        packet_loss = packet_loss[:T]

    assert len(packet_loss) == T, "Packet loss array length must match T."
    # print(packet_loss)
    return packet_loss

def system_dynamics(CONSTANTS, x, u, w):
    A = CONSTANTS["A"]
    B = CONSTANTS["B"]
    return A @ x + B @ u + w


def cost_function_info(Q_info, R_info, t, cycle_time):
    return Q_info[t % cycle_time], R_info[t % cycle_time]

# Cost function
def periodical_cost_function(t, x, u, Q_info, R_info, cycle_time):
    Qs, Rs = cost_function_info(Q_info, R_info, t, cycle_time)
    Q = np.diag([Qs, Qs])
    R = np.array([[1]])
    # Q = torch.tensor(Q, dtype=torch.float32)
    # R = torch.tensor(R, dtype=torch.float32)
    return x.T @ Q @ x + u.T @ R @ u

def periodical_cost_function_torch(t, x, u, Q_info, R_info, cycle_time):
    Qs, Rs = cost_function_info(Q_info, R_info, t, cycle_time)
    Q = np.diag([Qs, Qs])
    R = np.array([[1]])
    Q = torch.tensor(Q, dtype=torch.float32)
    R = torch.tensor(R, dtype=torch.float32)
    return x.T @ Q @ x + u.T @ R @ u

# Generate control input
def generate_control_input(K, x, Ms=None, ws=None):
    if Ms is None:
        return -K @ x
    else:
        return -K @ x + sum([m @ w for m, w in zip(Ms, ws)])

# Total cost calculation
def calculate_total_cost(CONSTANTS, K = None, T_max = None, use_history=False, cost_function=periodical_cost_function):
    T = CONSTANTS["T"]
    A = CONSTANTS["A"]
    B = CONSTANTS["B"]
    K = CONSTANTS["K"] if K is None else K
    p_data = CONSTANTS["PL_data"]
    w_data = CONSTANTS["pertubations"]
    Q_info = CONSTANTS["Q_info"]
    R_info = CONSTANTS["R_info"]
    cycle_time = CONSTANTS["cycle_time"]

    # if T_max is not None:
    #     T = min(T, T_max)

    x = np.array([0.0, 0.0])
    if use_history:
        x_history = []
        u_history = []
        cost_history = []
        # for t in tqdm.tqdm(range(T)):
        for t in range(T):
            u = generate_control_input(K, x) * p_data[t]
            cost = cost_function(t, x, u, Q_info, R_info, cycle_time)
            x = system_dynamics(CONSTANTS, x, u, w_data[t])
            x_history.append(x)
            u_history.append(u)
            cost_history.append(cost)

        return cost_history, x_history, u_history
    else:
        average_cost = 0
        # for t in tqdm.tqdm(range(T)):
        for t in range(T):
            # print(t, p_data[t])
            u = generate_control_input(K, x) * p_data[t]
            cost = cost_function(t, x, u, Q_info, R_info, cycle_time)
            x = system_dynamics(CONSTANTS, x, u, w_data[t])
            average_cost += cost/T
        return average_cost, [], []
    
def evaluate_system(CONSTANTS, A, B, K):
    eigen_values, eigen_vectors = scipy.linalg.eig(A - B @ K)
    gamma = max(np.abs(eigen_values))
    kappa_A = np.linalg.norm(A, 2)

    CONSTANTS["norm_A"] = kappa_A
    CONSTANTS["norm_ABK"] = np.linalg.norm(A - B @ K, 2)
    CONSTANTS["norm_K"] = max(np.linalg.norm(K, 2), np.linalg.norm(eigen_vectors, 2), np.linalg.norm(np.linalg.inv(eigen_vectors), 2))
    CONSTANTS["nu"] = gamma if kappa_A > 1 else kappa_A
    CONSTANTS["H"] = int(-np.log(CONSTANTS["T"])/np.log(CONSTANTS["nu"]))
    

def find_K(A, B, kappa, gamma, kappa_A):
    kappa2 = kappa / np.sqrt(2)    
    def objective(trial):
        K = np.array([[trial.suggest_float("K1", -kappa2, kappa2), trial.suggest_float("K2", -kappa2, kappa2)]])
        L, Q = scipy.linalg.eig(A - B @ K)
        kappa_Q = np.linalg.norm(Q, 2)
        kappa_Q_inv = np.linalg.norm(np.linalg.inv(Q), 2)
        norm_L = np.linalg.norm(L, 2)
        if kappa_Q >= kappa or kappa_Q_inv >= kappa or norm_L >= gamma or norm_L >= kappa_A:
            return 1e100

        return norm_L
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=1000)
    best_params = study.best_params
    best_K = np.array([[best_params['K1'], best_params['K2']]])
    return best_K


# Stable and unstable systems
def get_stable_system(CONSTANTS):
    kappa = CONSTANTS["kappa"]
    gamma = CONSTANTS["gamma"]
    A = np.array([[-0.4, 0.1], [0.2, -0.6]])
    B = np.array([[0.4], [0.3]])
    kappa_A = np.linalg.norm(A, 2)
    K = find_K(A, B, kappa, gamma, kappa_A)
    CONSTANTS["A"] = A
    CONSTANTS["B"] = B
    CONSTANTS["K"] = K
    evaluate_system(CONSTANTS, A, B, K)


def get_unstable_system(CONSTANTS):
    kappa = CONSTANTS["kappa"]
    gamma = CONSTANTS["gamma"]
    A = np.array([[2.0, 1.0], [1.0, 1.6]])
    B = np.array([[0.8], [0.6]])
    # K = np.array([[1.68391762, 3.31759806]])
    kappa_A = np.linalg.norm(A, 2)
    K = find_K(A, B, kappa, gamma, kappa_A)
    CONSTANTS["A"] = A
    CONSTANTS["B"] = B
    CONSTANTS["K"] = K
    evaluate_system(CONSTANTS, A, B, K)


import torch
def get_approximated_grad(CONSTANTS, M, p_history, w_history, Q, R):
    # M = torch.randn(H, 1, 2, requires_grad=True, dtype=torch.float32)
    # print(CONSTANTS)
    M.requires_grad_(True)
    A = torch.tensor(CONSTANTS["A"], dtype=torch.float32)
    B = torch.tensor(CONSTANTS["B"], dtype=torch.float32)
    K = torch.tensor(CONSTANTS["K"], dtype=torch.float32)
    H = CONSTANTS["H"]
    def approximated_function(M, A, B, K, w_history, p_history, Q, R):
        y = torch.zeros(2, 1)
        alpha = torch.eye(2)
        for i in range(H):
            y = y + alpha @ w_history[i + 1]
            tmp = torch.zeros(1)
            for j in range(H):
                tmp = tmp + M[j] @ w_history[i+j+2]
            # print("tmp shape:", tmp.shape, "\nB shape:", B.shape, "\n alpha shape:", alpha.shape)
            y = y + p_history[i+1] * alpha @ B @ tmp
            if i == H - 1:
                continue
            alpha = alpha @ (A - p_history[i+1] * B @ K)
        u = -K @ y
        for i in range(H):
            u = u + M[i] @ w_history[i+1]
        u = p_history[0] * u
        cost = y.T @ Q @ y + u.T @ R @ u
        return cost
    # Mについて勾配を計算
    cost = approximated_function(M, A, B, K, w_history, p_history, Q, R)
    cost.backward()
    # return M.grad
    grad = M.grad.clone()  # 勾配のコピーを取得
    M.grad.zero_()  # 勾配のリセット
    return grad

def update_M(M, grad, lr, norm_bound, nu):
    # norm_bound = 0.5
    # M_tmp = M - lr * grad
    # clip
    # print(torch.clamp(grad, max=1))
    M = M - lr * grad
    for i in range(M.shape[0]):
        # print("M_tmp:", M_tmp[i], "M:", M, "grad:", grad) 
        U, S, V = torch.svd(M[i])
        S = torch.clamp(S, max=norm_bound*nu**i)
        M[i] = U @ S @ V.T
    # return torch.tensor(M, dtype=torch.float32, requires_grad=True) 
    return M.clone().detach().requires_grad_(True)

def eta(W, T, t, initial=1):
    initial = 1
    # 簡易的に定数は無視して1からスタート
    return initial/(W * (np.sqrt(T)))# /(min(1 + 0.01*t, 10))
    # * 2**(int(10*t/T)))

from torch.utils.tensorboard import SummaryWriter

def adaptive_control(CONSTANTS):
    # TensorBoard Writerの初期化
    writer = SummaryWriter(log_dir=CONSTANTS["save_dir"])

    T = CONSTANTS["T"]
    nu = CONSTANTS["nu"]
    H = CONSTANTS["H"]

    M = torch.tensor(np.zeros((H, 1, 2)), dtype=torch.float32, requires_grad=True)
    A = torch.tensor(CONSTANTS["A"], dtype=torch.float32)
    B = torch.tensor(CONSTANTS["B"], dtype=torch.float32)
    K = torch.tensor(CONSTANTS["K"], dtype=torch.float32)
    p_data = torch.tensor(CONSTANTS["PL_data"], dtype=torch.float32)
    w_data = torch.tensor(CONSTANTS["pertubations"], dtype=torch.float32)
    # print(w_data)

    K_opt = torch.tensor(CONSTANTS["K_opt"], dtype=torch.float32)
    Q_info = CONSTANTS["Q_info"]
    R_info = CONSTANTS["R_info"]
    xi = 2 * np.linalg.norm(K, 2)**3
    kappa_B = np.linalg.norm(B, 2)

    cycle_time = CONSTANTS["cycle_time"]
    initial_eta = 1
    W = CONSTANTS["W"]

    previous_reg = 0
    reg_instant = 0
    sum_reg = 0
    cost = 0
    cost_opt = 0

    x = torch.zeros(2, 1)
    x_opt = torch.zeros(2, 1)


    # 不変の定数もwriterに記録
    writer.add_scalar("Constants/W", W)
    writer.add_scalar("Constants/T", T)
    writer.add_scalar("Constants/H", H)
    writer.add_scalar("Constants/nu", nu)
    writer.add_scalar("Constants/xi", xi)
    writer.add_scalar("Constants/norm_A", CONSTANTS["norm_A"])
    writer.add_scalar("Constants/norm_B", kappa_B)
    writer.add_scalar("Constants/norm_ABK", CONSTANTS["norm_ABK"])
    writer.add_scalar("Constants/initial_eta", initial_eta)
    writer.add_scalar("Constants/cycle_time", cycle_time)
    writer.add_scalar("Constants/kappa", CONSTANTS["kappa"])
    writer.add_scalar("Constants/gamma", CONSTANTS["gamma"])
    writer.add_scalar("Constants/K_opt", K_opt[0][0])
    writer.add_scalar("Constants/K", K[0][0])
    writer.add_scalar("Constants/eta", eta(W, T, 0, initial_eta))
    writer.add_scalar("Constants/rho", CONSTANTS["rho"])
    writer.add_scalar("Constants/f_G", CONSTANTS["f_G"])
    writer.add_scalar("Constants/f_L", CONSTANTS["f_L"])
    writer.add_scalar("Constants/n_G", CONSTANTS["n_G"])    


    for t in tqdm.tqdm(range(T)):
        writer.add_scalar("Pertubation/(x-w)_norm", torch.norm(x - w_data[t].reshape(-1, 1)).item(), t)
        # if p_data[t] == 0:
        writer.add_scalar("PacketLoss/p", p_data[t].item(), t)
        if t <= 2 * H:
            u = -torch.matmul(K, x)
            for i in range(H-1, -1, -1):
                if t - i - 1 < 0:
                    continue
                u = u + M[i] @ w_data[t - i - 1].reshape(-1, 1)
            cost = periodical_cost_function_torch(t, x, u, Q_info, R_info, cycle_time)

            x = torch.matmul(A, x) + torch.matmul(B, u) + w_data[t].reshape(-1, 1)
            Qs, Rs = cost_function_info(Q_info, R_info, t, cycle_time)
            Q = torch.tensor(np.diag([Qs, Qs]), dtype=torch.float32)
            R = torch.tensor(np.diag([Rs]), dtype=torch.float32)
            p_data_update = p_data[0:t+1]
            # 不足分を前から0で埋める
            p_data_update = torch.cat([torch.ones(2*H+1 - p_data_update.shape[0]), p_data_update], dim=0)
            w_data_update = w_data[0:t].reshape(t, 2, 1)
            w_data_update = torch.cat([torch.zeros(2*H+1 - w_data_update.shape[0], 2, 1), w_data_update], dim=0)
            grad = get_approximated_grad(CONSTANTS, M, p_data_update.flip(0), w_data_update.flip(0), Q, R)
            writer.add_scalar("Grad/Grad_norm", torch.norm(grad).item(), t)
            M = update_M(M, grad, eta(W, T, t, initial_eta), xi, nu)
            
        else:
            u = -torch.matmul(K, x)
            for i in range(H-1, -1, -1):
                u = u + M[i] @ w_data[t - i - 1].reshape(-1, 1)
            u = p_data[t] * u

            cost = periodical_cost_function_torch(t, x, u, Q_info, R_info, cycle_time)
            # cost_x.append(cost.item())
            x = torch.matmul(A, x) + torch.matmul(B, u) + w_data[t].reshape(-1, 1)
            Qs, Rs = cost_function_info(Q_info, R_info, t, cycle_time)
            Q = torch.tensor(np.diag([Qs, Qs]), dtype=torch.float32)
            R = torch.tensor(np.diag([Rs]), dtype=torch.float32)
            p_data_update = p_data[t-H-1:t]
            w_data_update = w_data[t-2*H-1:t].reshape(2*H+1, 2, 1)
            grad = get_approximated_grad(CONSTANTS, M, p_data_update.flip(0), w_data_update.flip(0),Q, R)
            writer.add_scalar("Grad/Grad_norm", torch.norm(grad).item(), t)
            M = update_M(M, grad, eta(W, T, t, initial_eta), xi, nu)
        
        u_opt = -p_data[t] * K_opt @ x_opt
        previous_cost_opt = cost_opt
        cost_opt = periodical_cost_function_torch(t, x_opt, u_opt, Q_info, R_info, cycle_time)
        # cost_x_opt.append(cost_opt.item())

        x_opt = torch.matmul(A, x_opt) + torch.matmul(B, u_opt) + w_data[t].reshape(-1, 1)

        # reg = (cost - cost_opt).item() # + (regret_history[-1] if len(regret_history) > 0 else 0)
        previous_reg = reg_instant
        reg_instant = (cost - cost_opt).item()
        sum_reg += reg_instant
        # regret_history.append(reg_instant)

        # TensorBoardにログを記録
        writer.add_scalar("States/x", torch.norm(x).item(), t)
        writer.add_scalar("States/x_opt", torch.norm(x_opt).item(), t)
        writer.add_scalar("ControlInput/u", torch.norm(u).item(), t)
        writer.add_scalar("ControlInput/u_opt", torch.norm(u_opt).item(), t)
        writer.add_scalar("Regret/Instant", reg_instant, t)
        writer.add_scalar("Cost/Cost_DAC", cost.item(), t)
        writer.add_scalar("Cost/Cost_Opt", cost_opt.item(), t)
        writer.add_scalar("States/Difference_Norm", torch.norm(x - x_opt).item(), t)
        writer.add_scalar("Regret/Cumulative", sum_reg, t)
        writer.add_scalar("Pertubation/W_norm", torch.norm(w_data[t]).item(), t)

    writer.close()
    return


def search_optimal_linear_policy(CONSTANTS):
    A = CONSTANTS["A"]
    B = CONSTANTS["B"]
    K = CONSTANTS["K"]
    T_max = 10000
    def objective(trial):
        K1 = trial.suggest_float("K1", -CONSTANTS["kappa"], CONSTANTS["kappa"])
        K2 = trial.suggest_float("K2", -CONSTANTS["kappa"], CONSTANTS["kappa"])
        K_candidate = np.array([[K1, K2]])

        # Check stability of A - BK
        L, Q = scipy.linalg.eig(A - B @ K_candidate)
        if np.max(np.abs(L)) >= 1 or K1**2 + K2**2 > CONSTANTS["kappa"]**2 or np.linalg.norm(Q, 2) >= CONSTANTS["kappa"] or np.linalg.norm(np.linalg.inv(Q), 2) >= CONSTANTS["kappa"]:
            return 1e100

        cost_history, _, _ = calculate_total_cost(CONSTANTS, K_candidate, T_max=T_max, use_history=False)
        # mean
        return cost_history

    # study = optuna.create_study()
    from joblib import parallel_backend
    from optuna.samplers import TPESampler

    study = optuna.create_study(sampler=TPESampler())

    study.enqueue_trial({"K1": K[0][0], "K2": K[0][1]})
    study.optimize(objective, n_trials=1000, n_jobs=1)

    # # Optimal results
    best_params = study.best_params
    best_K = np.array([[best_params['K1'], best_params['K2']]])

    CONSTANTS["K_opt"] = best_K

def searcher(W, T, n_G, rho, name="trial"):
    # W = 3
    # fg = 1
    # T = 10000
    # n_l = 1
    # l_L = 2* int(np.log(T))

    Q_info = [1.0, 1.0, 0.4, 0.4, 0.2, 0.2, 0.6, 0.6]
    R_info = [1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0, 1,0]
    cycle_time = 8
    CONSTANTS = {
        "save_dir": f"results/{name}", #{W}_{T}_{f_G}_{n_l}_{l_L}_{eta_init}",
        "W": W,
        "T": T,
        "l_L": l_L,
        "n_G": n_G,
        "f_G": n_G/T,
        "rho": rho,
        "eta": eta(W, T, 0, eta_init),
        "Q_info": Q_info,
        "R_info": R_info,
        "cycle_time": cycle_time,
        "kappa": 10.0,
        "gamma": 0.9,
    }
    os.makedirs(CONSTANTS["save_dir"], exist_ok=True)
    # get_stable_system(CONSTANTS)
    get_unstable_system(CONSTANTS)

    H = CONSTANTS["H"]
    f_L = rho/H
    CONSTANTS["f_L"] = f_L


    

    # pertubation_path = f"pertubations/{T}_{W}_pertubations.npy"
    # PL_data_path = f"PL_data/{T}_{CONSTANTS['f_G']}_{CONSTANTS['f_L']}_{CONSTANTS['l_L']}.npy"
    # if not os.path.exists(pertubation_path):
    # # 一度に生成する長さ
    pertubations = []
    pertubation = np.array([0, 0])
    for _ in range(T):
        pertubation = generate_perturbation_in_advance(pertubation, W)
        pertubations.append(pertubation)
    #     # ファイルに保存
    #     np.save(pertubation_path, pertubations)
    # else:
    #     pertubations = np.load(pertubation_path)
    
    CONSTANTS["pertubations"] = pertubations
    print("Pertubations generated")
    # if not os.path.exists(PL_data_path):
    PL_data = generate_packetloss_in_advance(T, CONSTANTS["f_G"], CONSTANTS["rho"], CONSTANTS["H"])

    # np.save(PL_data_path, PL_data)
    # else:
        # PL_data = np.load(PL_data_path)
    # print(len(PL_data[PL_data==0]))

    # # 観察のために、1000, 1500, 2000, 2500, 3000,あたりにn_k個の連続PLを入れる
    # PL_data[1000:1000+n_l] = 0
    # PL_data[1500:1500+n_l] = 0
    # PL_data[2000:2000+n_l] = 0
    # PL_data[2500:2500+n_l] = 0
    # PL_data[3000:3000+n_l] = 0

    
    CONSTANTS["PL_data"] = PL_data
    print("Packet loss data generated")

    search_optimal_linear_policy(CONSTANTS)
    # K_opt = np.array([[0.39429104504938967, 0.22964920215856816]])
    # K_opt = np.array([[-0.05110982, -0.08169227]]) # stable
    # K_opt = np.array([[1.92265214, 1.59026545]]) # unstable
    # CONSTANTS["K_opt"] = K_opt
    print("Optimal policy found")
    adaptive_control(CONSTANTS)


# Main script
if __name__ == "__main__":
    # search_area = {
    #     "W": [1, 2, 4, 8, 16],
    #     "T": [4000, 4000, 8000, 16000, 32000],
    #     "n_L": [1, 2, 4, 8],
    #     "f_G": ["log", "sqrt", "linear"],
    #     "fg": [1, 2, 4, 8, 16],
    #     "eta_init": [1e-2, 1e-1, 1, 1e1, 1e2]
    # }

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--search", action="store_true")
    # parser.add_argument("--trial", action="store_true")
    # parser.add_argument("--case", type=int, default=1)
    # # parser.add_argment("-W", type=int, default=1)
    # # parser.add_argment("-T", type=int, default=10000)
    # # parser.add_argment("-n_l", type=int, default=1)
    # # parser.add_argment("-fg", type=int, default=1)
    # args = parser.parse_args()
    # search = args.search

    search_area = {
        "W": [4],
        "T": [64000],
        "n_L": [1],
        "f_G": ["sqrt"],
        "fg": [4],
        "eta_init": [1]
    }
    # search_area = {
    #     "W": [1],
    #     "T": [1000],
    #     "n_L": [1],
    #     "f_G": ["sqrt"],
    #     "fg": [1]
    # }
    # from concurrent.futures import ProcessPoolExecutor
    # with ProcessPoolExecutor() as executor:
    #     futures = []
    #     for W in search_area["W"]:
    #         for T in search_area["T"]:
    #             for n_L in search_area["n_L"]:
    #                 for f_G in search_area["f_G"]:
    #                     for eta_init in search_area["eta_init"]:
    #                         for fg in search_area["fg"]:
    #                             if f_G == "log":
    #                                 f_G = fg/np.log(T)
    #                             elif f_G == "sqrt":
    #                                 f_G = fg/np.sqrt(T)
    #                             elif f_G == "linear":
    #                                 f_G = fg/T
    #                             # searcher(W, T, f_G, n_L, 2*int(np.log(T)))
    #                             futures.append(executor.submit(searcher, W, T, f_G, n_L, 2*int(np.log(T)), eta_init))
    #     for future in tqdm.tqdm(futures):
    #         future.result()
    # searcher(1, 10000, 1, 1, 2*int(np.log(10000)))
    # searcher(4, 64000, 4/np.sqrt(64000), 1, 2*int(np.log(64000)), 1)
    W = 4
    T = 100000
    name = "trial_ex_unstable_50" # 4はupdateを常時
    # f_G = 1/(np.sqrt(T))
    n_G = int(np.sqrt(T))
    rho = 2
    l_L = 2*int(np.log(T))
    eta_init = 1
    searcher(W, T, n_G, rho, name)


    # for unstable
    T_search_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
    rho_search_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # fix T = 100000 -> H = 13
    W_search_list = [0.1, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    standard_T = 100000
    standard_rho = 4
    standard_W = 4
    tasks = []

    for T in T_search_list:
        tasks.append((T, standard_W, standard_rho))
    for rho in rho_search_list:
        tasks.append((standard_T, standard_W, rho))
    for W in W_search_list:
        tasks.append((standard_T, W, standard_rho))

    from multiprocessing import Pool
    with Pool(18) as p:
        p.map(searcher, tasks)








    # A: stable -> W, T, f_Gについての実験
    # A: unstable -> W, T, l_Lについての実験


    

    # if args.trial:
    #     pertubations = np.load(f"{T}_pertubations.npy")
    #     PL_data = generate_packetloss_in_advance(T, CONSTANTS["f_G"], CONSTANTS["f_L"], CONSTANTS["l_L"])
    #     K_opt = np.array([[0.17008674, 0.18522483]])
    #     # CONSTANTS = evaluate_system(A, B, K)
    #     CONSTANTS["H"] = int(-np.log(T)/np.log(CONSTANTS["nu"]))

    #     # regret_history = adaptive_control(A, B, K, K_opt, T, PL_data, pertubations, Q_info, R_info, cycle_time, H, initial_eta, start_vis, CONSTANTS["nu"], W)
    #     regret_history = adaptive_control()
    #     plt.plot(regret_history)

    # else:
    #     if search:
            

    #     else:
    #         # Test optimal control
    #         K = CONSTANTS["K"]
    #         pertubations = np.load(f"{T}_pertubations.npy")
    #         PL_data = np.load(f"PL_data_{CONSTANTS['T']}_{CONSTANTS['f_G']}_{CONSTANTS['f_L']}_{CONSTANTS['l_L']}.npy")
    #         # cost_history, x_history, u_history = calculate_total_cost(T, A, B, PL_data, pertubations, K, np.array([0.0, 0.0]), Q_info, R_info, cycle_time, use_history=True)
    #         cost_history, x_history, u_history = calculate_total_cost(use_history=True)
    #         H = int(-np.log(T)/np.log(CONSTANTS["nu"]))

    #         # loss_indices = (np.array(PL_data) == 0)
    #         # loss indices pl=0の次のindexを取得、Tより大きい要素は除外
    #         loss_indices = (np.array(PL_data) == 0)
    #         loss_indices = np.roll(loss_indices, 1)
    #         loss_indices = loss_indices[loss_indices < T]

    #         time_steps = np.arange(T)
    #         # cost履歴のplot
    #         plt.plot(np.log1p(cost_history))
    #         # pl = 0の箇所のみ赤でscatter
    #         plt.scatter(np.where(loss_indices)[0], np.log1p(cost_history)[loss_indices], c="red", label="Packet loss")
    #         plt.xlabel("Time step")
    #         plt.ylabel("Cost")
    #         plt.title("Cost history")
    #         plt.savefig("cost_history.png")
    #         plt.clf()

    #         # cost 蓄積のplot
    #         accumulated_cost = np.cumsum(cost_history)
    #         # plt.plot(np.log1p(time_steps), np.log1p(accumulated_cost))
    #         plt.plot(time_steps, accumulated_cost)
    #         # pl = 0の箇所のみ赤でscatter
    #         # plt.scatter(np.log1p(time_steps)[loss_indices], np.log1p(accumulated_cost)[loss_indices], c="red", label="Packet loss")
    #         plt.scatter(time_steps[loss_indices], accumulated_cost[loss_indices], c="red", label="Packet loss")
    #         plt.xlabel("Time step")
    #         plt.ylabel("Accumulated cost")
    #         plt.title("Accumulated cost history")
    #         plt.savefig("accumulated_cost_history.png")
    #         plt.clf()


    #         # 状態履歴のplot
    #         x_history = np.array(x_history)
    #         plt.plot(x_history[:, 0], x_history[:, 1], label="x1-x2")
    #         # pl = 0の箇所のみ赤でscatter
    #         plt.scatter(x_history[loss_indices, 0],x_history[loss_indices, 1], c="red", label="Packet loss")
    #         plt.xlabel("x1")
    #         plt.ylabel("x2")
    #         plt.title("State history")
    #         plt.savefig("state_history.png")
    #         plt.clf()

    #         # 制御入力履歴のplot
    #         u_history = np.array(u_history)
    #         plt.plot(u_history)
    #         # pl = 0の箇所のみ赤でscatter
    #         plt.scatter(np.where(loss_indices)[0], u_history[loss_indices], c="red", label="Packet loss")
    #         plt.xlabel("Time step")
    #         plt.ylabel("Control input")
    #         plt.title("Control input history")
    #         plt.savefig("control_input_history.png")
    #         plt.clf()

    #         print("Total cost:", np.sum(cost_history))
    #         print("Average cost:", np.mean(cost_history))
    #         print("Max cost:", np.max(cost_history))
    #         print("Min cost:", np.min(cost_history))