import argparse
import json
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
        
        phi = 2 * self.l_q * self.lambda_q * m_i**2 * (H_bar - self.H)**2
        p_lower = self.c - self.alpha * phi / (m_i**3 + epsilon)
        p_upper = self.c - self.alpha * phi / ((m_i + self.alpha)**3 + epsilon)
        # print("client", self.id, "p_lower", p_lower, "p_upper", p_upper, "p", p)
        if p <= p_lower:
            return 0
        elif p >= p_upper:
            return 1
        else:
            if abs(self.c - p) < epsilon:
                return 0  # If p is very close to c, client doesn't participate
            x = (phi / (self.alpha**2 * max(self.c - p, epsilon)))**(1/3) - m_i / self.alpha
            return max(min(x, 1), 0)

class Server:
    def __init__(self, clients, lambda_v, lambda_s, l_v, l_s, H_N_bar, H_O_bar, B):
        self.clients = clients
        self.lambda_v = lambda_v
        self.lambda_s = lambda_s
        self.l_v = l_v
        self.l_s = l_s
        self.H_N_bar = H_N_bar
        self.H_O_bar = H_O_bar
        self.B = B
        self.H_star = (self.lambda_v * self.l_v * self.H_N_bar + self.lambda_s * self.l_s * self.H_O_bar) / (self.lambda_v * self.l_v + self.lambda_s * self.l_s)

    def calculate_H(self, x):
        numerator = sum(client.alpha * x[i] * client.H for i, client in enumerate(self.clients))
        denominator = sum(client.alpha * x[i] for i, client in enumerate(self.clients))
        return numerator / denominator if denominator != 0 else self.H_N_bar

    def utility(self, p):
        x = self.get_client_strategies(p)
        H = self.calculate_H(x)
        u_s = self.lambda_v * self.l_v * (H - self.H_N_bar)**2 + self.lambda_s * self.l_s * (H - self.H_O_bar)**2
        return u_s

    def get_client_strategies(self, p):
        H_bar = sum(client.alpha * client.H for client in self.clients)
        x = np.random.rand(len(self.clients))  # Random initial strategy
        for _ in range(100):  # Increase iterations for better convergence
            old_x = x.copy()
            total_alpha_x = sum(client.alpha * x[i] for i, client in enumerate(self.clients))
            for i, client in enumerate(self.clients):
                m_i = total_alpha_x - client.alpha * x[i]
                x[i] = client.best_response(p[i], H_bar, m_i)
            if np.allclose(x, old_x, atol=1e-4):
                break
        return x

    # def constraint_H(self, p):
    #     x = self.get_client_strategies(p)
    #     return self.H_star - self.calculate_H(x)

    def constraint_budget(self, p):
        x = self.get_client_strategies(p)
        return self.B - sum(p[i] * x[i] for i in range(len(self.clients)))

    def solve(self):
        bounds = [(0, None) for _ in self.clients]
        constraints = [
            # {'type': 'ineq', 'fun': self.constraint_H},
            {'type': 'ineq', 'fun': self.constraint_budget}
        ]
        
        initial_p = [1.5 * client.c for client in self.clients]  # Start with payments slightly higher costs
        result = minimize(self.utility, initial_p, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-3, 'maxiter': 1000})
        
        optimal_p = result.x
        optimal_x = self.get_client_strategies(optimal_p)
        
        return optimal_p, optimal_x

def binary_search_budget(server, low, high, tolerance=0.01):
    while high - low > tolerance:
        mid = (low + high) / 2
        server.B = mid
        optimal_p, optimal_x = server.solve()
        full_participants = sum(1 for x in optimal_x if x > 0)  # Count clients with x > 0.99 as full participants
        
        if full_participants > int(2 * len(server.clients) / 3):
            high = mid
        elif full_participants <= int(len(server.clients) / 3) + 1:
            low = mid
        else:
            return mid  # We've found a budget in the desired range
    
    return (low + high) / 2


def main(args):
    dataset_name = args.dataset
    num_clients = args.num_clients
    alpha = args.alpha
    
    with open(f'partitions/partition_indices_{dataset_name}_clients{num_clients}_alpha{alpha}/wasserstein_distances_{dataset_name}_alpha{alpha}.json', 'r') as f:
        results = json.load(f)

    H_N_bar = results["H_N"]
    H_O_bar = results["H_O"]
    weights = []
    Hs = []
    idx_hash_client = {}
    idx = 0
    for client, distance in results["wasserstein_distances"].items():
        client_id = int(client)
        idx_hash_client[idx] = client_id
        idx += 1
        weights.append(results["weights"][client_id])
        Hs.append(distance)
    
    weights = np.array(weights)/np.sum(weights)
    
    # Set up hyperparameters and create clients
    N = len(Hs)
    lambda_v, lambda_s = 1.0, 1.0
    l_v = l_q = l_s = 1e4
    lambda_q = 1.0


    np.random.seed(42) 
    costs = np.random.uniform(0.5, 1.5, N)
    clients = [
        Client(i, weights[i], Hs[i], costs[i], l_q, lambda_q)
        for i in range(N)
    ]

    # Create server with initial budget
    initial_budget = sum(client.c for client in clients)  # Start with a budget equal to the sum of all costs
    server = Server(clients, lambda_v, lambda_s, l_v, l_s, H_N_bar, H_O_bar, initial_budget)

    # Search for the appropriate budget
    optimal_budget = binary_search_budget(server, 0, initial_budget * 2)

    # Solve the Stackelberg game with the optimal budget
    server.B = optimal_budget
    optimal_p, optimal_x = server.solve()

    # Print results
    print(f"Optimal Budget: {optimal_budget:.4f}")

    print("\nOptimal payments (p):")
    for i, p in enumerate(optimal_p):
        print(f"Client {i}: {p:.4f}")

    print("\nOptimal strategies (x):")
    fu_clients = []
    for i, x in enumerate(optimal_x):
        print(f"Client {i}: {x:.4f}")
        if x > 0:
            fu_clients.append(idx_hash_client[i])

    print(f"\nFinal utility: {server.utility(optimal_p):.4f}")
    print(f"Final H: {server.calculate_H(optimal_x):.4f}")
    print(f"H_star: {server.H_star:.4f}")
    print(f"Budget used: {sum(p * x for p, x in zip(optimal_p, optimal_x)):.4f}")

    # Print client details
    print("\nClient details:")
    for i, client in enumerate(clients):
        print(f"Client {i}: alpha={client.alpha:.4f}, H={client.H:.4f}, c={client.c:.4f}")

    # # Count full participants
    # full_participants = sum(1 for x in optimal_x if x > 0)
    # print(f"\nNumber of participants (x > 0): {full_participants} out of {N}")
    
    
    results["game_results"] = {
        "optimal_budget": optimal_budget,
        "optimal_payments": optimal_p.tolist(),
        "optimal_strategies": optimal_x.tolist(),
        "final_utility": server.utility(optimal_p),
        "final_H": server.calculate_H(optimal_x),
        "H_star": server.H_star,
        "budget_used": sum(p * x for p, x in zip(optimal_p, optimal_x)),
        "fu_clients": fu_clients
    }
    print("results", results)
    # save results
    with open(f'partitions/partition_indices_{dataset_name}_clients{num_clients}_alpha{alpha}/wasserstein_distances_{dataset_name}_alpha{alpha}.json', 'w') as f:
        json.dump(results, f, indent=2)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Wasserstein distances for federated learning client partitions.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (mnist or cifar10)')
    parser.add_argument('--num_clients', type=int, required=True, help='Number of clients')
    parser.add_argument('--alpha', type=float, required=True, help='Dirichlet distribution alpha parameter')

    args = parser.parse_args()
    main(args)