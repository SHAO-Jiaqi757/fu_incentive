import argparse
import json
from typing import List
import numpy as np
from scipy.optimize import minimize


class Client:
    def __init__(self, id, alpha, H, c, l_q, lambda_q):
        self.id = id
        self.alpha = alpha
        self.H = H
        self.c = c
        self.l_q = l_q
        self.lambda_q = lambda_q

    def best_response(self, p, H_bar, m_i):
        epsilon = 1e-10  # Small value to avoid division by zero

        if m_i < epsilon:
            return 0  # If m_i is close to zero, client doesn't participate

        phi = 2 * self.l_q * self.lambda_q * (H_bar) ** 2
        p_lower = self.c - self.alpha * phi / (m_i**3 + epsilon)
        p_upper = self.c - self.alpha * phi / ((m_i + self.alpha) ** 3 + epsilon)
        # print("client", self.id, "p_lower", p_lower, "p_upper", p_upper, "p", p)
        if p <= p_lower:
            return 0
        elif p >= p_upper:
            return 1
        else:
            if abs(self.c - p) < epsilon:
                return 0  # If p is very close to c, client doesn't participate
            x = (phi / (self.alpha**2 * max(self.c - p, epsilon))) ** (
                1 / 3
            ) - m_i / self.alpha
            return max(min(x, 1), 0)


class Server:
    def __init__(self, clients, lambda_v, lambda_s, l_v, l_s, H_N_bar, H_O_bar, unified_payment):
        self.clients = clients
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s
        self.l_v = l_v
        self.l_s = l_s
        self.H_N_bar = H_N_bar
        self.H_O_bar = H_O_bar
        self.unified_payment = unified_payment
        self.H_star = (
            self.lambda_v * self.l_v * self.H_N_bar
            + self.lambda_s * self.l_s * self.H_O_bar
        ) / (self.lambda_v * self.l_v + self.lambda_s * self.l_s)

    def calculate_H(self, x):
        numerator = sum(
            client.alpha * x[i] * client.H for i, client in enumerate(self.clients)
        )
        denominator = sum(client.alpha * x[i] for i, client in enumerate(self.clients))
        return numerator / denominator if denominator != 0 else self.H_N_bar

    def utility(self, p):
        x = self.get_client_strategies(p)
        H = self.calculate_H(x)
        u_s = (
            self.lambda_v * self.l_v * (H - self.H_N_bar) ** 2
            + self.lambda_s * self.l_s * (H - self.H_O_bar) ** 2
        )
        return u_s

    def get_client_strategies(self, p):
        
        x = np.random.rand(len(self.clients))  # Random initial strategy
        for _ in range(100):  # Increase iterations for better convergence
            old_x = x.copy()
            H_bar = sum(client.alpha * client.H * old_x[i] for i, client in enumerate(self.clients))
            total_alpha_x = sum(
                client.alpha * x[i] for i, client in enumerate(self.clients)
            )
            for i, client in enumerate(self.clients):
                H_bar = H_bar - sum(o_client.alpha * client.H * old_x[i] for i, o_client in enumerate(self.clients))
                m_i = total_alpha_x - client.alpha * x[i]
                x[i] = client.best_response(p[i], H_bar, m_i)
            if np.allclose(x, old_x, atol=1e-4):
                break
        return x

    def solve(self):
        optimal_x = self.get_client_strategies(self.unified_payment)

        return self.unified_payment, optimal_x


def cal_utility_clients(
    server: Server,
    p: List[float],
    x: List[float],
    Hs: List[float],
    costs: List[float],
    H_O_bar: float,
    l_q=1,
    lambda_q=1,
):
    utilities = []
    for i, H_i in enumerate(Hs):
        payment = p[i]
        best_response = x[i]

        Q_i = l_q * ((server.calculate_H(x) - H_i) ** 2 - (H_O_bar - H_i) ** 2)

        utility = payment * best_response - costs[i] * best_response - lambda_q * Q_i
        utilities.append(utility)
    return utilities

def main(args):
    dataset_name = args.dataset
    num_clients = args.num_clients
    alpha = args.alpha

    statistics_file = f"partitions/partition_indices_{dataset_name}_clients{num_clients}_alpha{alpha}/statistics_lambda_v{args.lambda_v}_lambda_s{args.lambda_s}_lambda_q{args.lambda_q}.json"
    with open(statistics_file, "r") as f:
        results = json.load(f)

    H_N_bar = results["H_N"]
    H_O_bar = results["H_O"]
    weights = []
    Hs = []
    removed_clients = results["removed_clients"]
    remain_clients_num = num_clients - len(removed_clients)
    budget = results["game_results"]["budget_used"]
    unified_payment = [budget / remain_clients_num] * remain_clients_num
    idx_hash_client = {}
    idx = 0
    for client, distance in results["wasserstein_distances"].items():
        client_id = int(client)
        idx_hash_client[idx] = client_id
        idx += 1
        weights.append(results["weights"][client_id])
        Hs.append(distance)

    weights = np.array(weights) / np.sum(weights)

    # Set up hyperparameters and create clients
    N = len(Hs)
    l_v = l_q = l_s = 1e1

    np.random.seed(42)
    costs = weights * 10  # Costs are proportional to weights
    clients = [
        Client(i, weights[i], Hs[i], costs[i], l_q, args.lambda_q) for i in range(N)
    ]

   
    server = Server(
        clients,
        args.lambda_v,
        args.lambda_s,
        l_v,
        l_s,
        H_N_bar,
        H_O_bar,
        unified_payment
    )

    optimal_p, optimal_x = server.solve()

    # Print results

    print("\nOptimal payments (p):")
    for i, p in enumerate(optimal_p):
        print(f"{idx_hash_client[i]}: {p:.4f}")

    print("\nOptimal strategies (x):")
    fu_clients = []
    for i, x in enumerate(optimal_x):
        print(f"Client {idx_hash_client[i]}: {x:.4f}")
        if x > 0.1:
            fu_clients.append(idx_hash_client[i])

    print(f"\nFinal utility: {server.utility(optimal_p):.4f}")
    print(f"Final H: {server.calculate_H(optimal_x):.4f}")
    print(f"H_star: {server.H_star:.4f}")
    print(f"Budget used: {sum(p * x for p, x in zip(optimal_p, optimal_x)):.4f}")

    # Print client details
    print("\nClient details:")
    for i, client in enumerate(clients):
        print(
            f"Client {idx_hash_client[i]}: alpha={client.alpha:.4f}, H={client.H:.4f}, c={client.c:.4f}"
        )

    # # Count full participants
    # full_participants = sum(1 for x in optimal_x if x > 0)
    # print(f"\nNumber of participants (x > 0): {full_participants} out of {N}")

    utility_clients = cal_utility_clients(
        server, optimal_p, optimal_x, Hs, costs, H_O_bar, l_q, args.lambda_q
    )

    results["unified_p_game_results"] = {
        "optimal_payments": optimal_p,
        "optimal_strategies": optimal_x.tolist(),
        "final_utility": server.utility(optimal_p),
        "final_H": server.calculate_H(optimal_x),
        "H_star": server.H_star,
        "budget_used": sum(p * x for p, x in zip(optimal_p, optimal_x)),
        "fu_clients": fu_clients,
        "utility_clients": utility_clients,
        "lambda_v": args.lambda_v,
        "lambda_s": args.lambda_s,
        "lambda_q": args.lambda_q,
    }
    print("results", results)
    # save results
    with open(statistics_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Wasserstein distances for federated learning client partitions."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (mnist, cifar10, cifar100)",
    )
    parser.add_argument(
        "--num_clients", type=int, required=True, help="Number of clients"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Dirichlet distribution alpha parameter",
    )
    parser.add_argument(
        "--lambda_v", type=float, default=1.0, help="lambda_v hyperparameter"
    )
    parser.add_argument(
        "--lambda_s", type=float, default=1.0, help="lambda_s hyperparameter"
    )
    parser.add_argument(
        "--lambda_q", type=float, default=1.0, help="lambda_q hyperparameter"
    )

    args = parser.parse_args()
    main(args)
