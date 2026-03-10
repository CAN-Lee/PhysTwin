
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_constitutive_comparison():
    # Setup parameters
    E = 1e5
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    fiber_k = 2e5
    
    # Range of stretch lambda
    lambdas = np.linspace(0.5, 2.0, 100)
    
    # Store results
    p_nh = []
    p_co = []
    p_st = []
    p_fi = []
    
    for l in lambdas:
        # 1D Deformation Gradient F = diag(l, 1, 1)
        F = torch.tensor([[l, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        I = torch.eye(3)
        
        # 1. Neo-Hookean
        detF = torch.det(F)
        FinvT = torch.inverse(F).t()
        lnJ = torch.log(detF)
        P_nh = mu * (F - FinvT) + lam * lnJ * FinvT
        p_nh.append(P_nh[0, 0].item())
        
        # 2. Corotated
        U, S, Vh = torch.linalg.svd(F)
        R = torch.matmul(U, Vh)
        J = torch.det(F)
        P_co = 2 * mu * (F - R) + lam * (J - 1) * J * FinvT
        p_co.append(P_co[0, 0].item())
        
        # 3. StVK
        E_green = 0.5 * (torch.matmul(F.t(), F) - I)
        trE = torch.trace(E_green)
        S_stvk = 2 * mu * E_green + lam * trE * I
        P_st = torch.matmul(F, S_stvk)
        p_st.append(P_st[0, 0].item())
        
        # 4. Fiber (direction [1, 0, 0])
        d = torch.tensor([1.0, 0, 0])
        Fd = torch.matmul(F, d)
        l_fib = torch.norm(Fd)
        if l_fib > 1.0:
            P_fi = fiber_k * (l_fib - 1) / l_fib * torch.matmul(F, torch.ger(d, d))
        else:
            P_fi = torch.zeros(3, 3)
        p_fi.append(P_fi[0, 0].item())

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, p_nh, label='Neo-Hookean', linewidth=2)
    plt.plot(lambdas, p_co, label='Corotated', linewidth=2)
    plt.plot(lambdas, p_st, label='StVK', linewidth=2, linestyle='--')
    plt.plot(lambdas, p_fi, label='Anisotropic Fiber', linewidth=2, color='red')
    
    # Mixture Example
    p_mix = 0.3 * np.array(p_nh) + 0.1 * np.array(p_co) + 0.1 * np.array(p_st) + 0.5 * np.array(p_fi)
    plt.plot(lambdas, p_mix, label='Mixture (Example)', linewidth=3, color='black', alpha=0.7)
    
    plt.axhline(0, color='black', lw=1)
    plt.axvline(1, color='gray', linestyle=':', lw=1)
    plt.title('Stress-Stretch Comparison of Constitutive Experts', fontsize=14)
    plt.xlabel('Stretch $\lambda$', fontsize=12)
    plt.ylabel('First Piola-Kirchhoff Stress $P_{11}$', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = 'assets/constitutive_comparison.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path} (and .png)")

def plot_simplified_diagram():
    # Setup parameters (reuse same logic as above or simplify)
    E = 1e5
    nu = 0.3
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    fiber_k = 2e5
    
    lambdas = np.linspace(0.5, 2.0, 100)
    p_nh, p_co, p_st, p_fi = [], [], [], []
    
    for l in lambdas:
        F = torch.tensor([[l, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        I = torch.eye(3)
        
        # 1. Neo-Hookean
        detF = torch.det(F)
        FinvT = torch.inverse(F).t()
        lnJ = torch.log(detF)
        P_nh = mu * (F - FinvT) + lam * lnJ * FinvT
        p_nh.append(P_nh[0, 0].item())
        
        # 2. Corotated
        U, S, Vh = torch.linalg.svd(F)
        R = torch.matmul(U, Vh)
        J = torch.det(F)
        P_co = 2 * mu * (F - R) + lam * (J - 1) * J * FinvT
        p_co.append(P_co[0, 0].item())
        
        # 3. StVK
        E_green = 0.5 * (torch.matmul(F.t(), F) - I)
        trE = torch.trace(E_green)
        S_stvk = 2 * mu * E_green + lam * trE * I
        P_st = torch.matmul(F, S_stvk)
        p_st.append(P_st[0, 0].item())
        
        # 4. Fiber
        d = torch.tensor([1.0, 0, 0])
        Fd = torch.matmul(F, d)
        l_fib = torch.norm(Fd)
        if l_fib > 1.0:
            P_fi = fiber_k * (l_fib - 1) / l_fib * torch.matmul(F, torch.ger(d, d))
        else:
            P_fi = torch.zeros(3, 3)
        p_fi.append(P_fi[0, 0].item())

    # Simplified Plotting
    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, p_nh, label='Neo-Hookean', linewidth=6)
    plt.plot(lambdas, p_co, label='Corotated', linewidth=6)
    plt.plot(lambdas, p_st, label='StVK', linewidth=6, linestyle='--')
    plt.plot(lambdas, p_fi, label='Anisotropic Fiber', linewidth=6, color='red')
    
    plt.axhline(0, color='black', lw=1)
    plt.axvline(1, color='gray', linestyle=':', lw=1)
    
    # Remove ticks and labels
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Stretch $\lambda$', fontsize=22)
    plt.ylabel('Stress $P$', fontsize=22)
    
    plt.title('Constitutive laws', fontsize=28)
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.1)
    
    output_path = 'assets/constitutive_models_simple.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Simplified plot saved to {output_path} (and .png)")

if __name__ == "__main__":
    plot_constitutive_comparison()
    plot_simplified_diagram()
