"""
Supplementary code for the article:

AlphaFold’s Bayesian Roots in Probability Kinematics
AISTATS 2026
Thomas Hamelryck & Kanti V. Mardia
"""
import torch

# Pyro deep probabilistic programming language - www.pyro.ai
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_median

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, ks_1samp
import random


###################################################################################
#                     === Configurable Parameters ===                             #
###################################################################################
N = 5                             # Number of steps in the random walk
kappa_prior = 5.0                 # Concentration of von Mises prior
alpha_beta = 10.0                 # Beta target alpha
beta_beta = 10.0                  # Beta target beta
num_samples = 1000                # Number of MCMC samples (>200)
burnin_steps = 1000               # MCMC warmup steps
n_trajectories = 500              # Number of trajectories to plot (<= num_samples)
###################################################################################

max_R = float(N)                  # Max possible resultant length
EPS = 1e-7
assert n_trajectories <= num_samples
assert num_samples > 200


PRINT_VERSIONS=0
if PRINT_VERSIONS:
    # Print the version numbers of used packages
    import matplotlib
    import scipy
    print("Torch ", torch.__version__)
    print("Pyro ", pyro.__version__)
    print("Matplotlib ", matplotlib.__version__)
    print("Numpy ", np.__version__)
    print("Scipy ", scipy.__version__)


def set_seed(SEED):
    """
    Seed for reproducibility

    SEED: int
    """
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


def stephens_logpdf(R, N, kappa):
    """
    Stephens (1969) approximation for the prior over resultant length R. 
    This will be used in the denominator of the PK update.

    R: value of length X0 to XN
    N: number of steps in the random walk
    kappa: concentration of von Mises distribution in random walk

    Returns: log of the PDF of R given by Stephens (1969)
    """
    gamma_star = 1.0 / (1.0 / kappa + 3.0 / (8.0 * kappa**2))
    x = 2 * N * gamma_star * (1 - R / N)
    chi2_dist = dist.Chi2(df=N - 1)
    log_pdf = chi2_dist.log_prob(x)
    return torch.log(torch.tensor(2 * gamma_star)) + log_pdf


def beta_logpdf_scaled(R, alpha, beta, max_R):
    """
    Scaled Beta target log PDF.
    This will be used in the numerator of the PK update.

    R: value of length X0 to XN
    alpha, beta: parameters of the Beta distribution
    max_R: maximum length of R

    Returns: log of the PDF of the scaled Beta distribution
    """
    R_scaled = R / max_R
    beta_dist = dist.Beta(alpha, beta)
    return beta_dist.log_prob(R_scaled) - torch.log(torch.tensor(max_R, dtype=torch.float64))


def model_rrm_vm(use_reference=True):
    """
    Probabilistic model implemented in Pyro (www.pyro.ai).

    use_reference: if False, the reference distribution is omitted (used in Ablation)

    returns: sampled angles
    """
    # Sample the angles from the von Mises
    theta = pyro.sample("theta", dist.VonMises(0.0, kappa_prior).expand([N]).to_event(1))
    # Calculate resultant length - distance of endpoint to origin
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    C = torch.mean(cos_theta)
    S = torch.mean(sin_theta)
    # Resultant vector length
    R = N * torch.sqrt(C**2 + S**2)

    # Numerator of PK update
    log_p = beta_logpdf_scaled(R, alpha=alpha_beta, beta=beta_beta, max_R=max_R)
    # Denominator of PK update
    if use_reference:
        # Conventional PK update with reference distribution
        log_pi = stephens_logpdf(R, N=N, kappa=kappa_prior)
        # Add small value to avoid underflows
        log_ratio = torch.log(torch.exp(log_p) + EPS) - torch.log(torch.exp(log_pi) + EPS)
    else:
        # Ablation without reference distribution
        log_ratio = torch.log(torch.exp(log_p) + EPS) 
    # Add the ratio target/reference to the density we sample from
    pyro.factor("pk_ratio", log_ratio)
    return theta


def run_inference(use_reference):
    """
    We sample from the Pyro model using Hamiltonian Monte Carlo / NUTS sampling.

    use_reference: if False, the reference distribution is omitted (used in Ablation)

    rteurns: MCMC samples of the angles
    """
    nuts_kernel = NUTS(model_rrm_vm)
    mcmc = MCMC(nuts_kernel, 
            num_samples=num_samples, 
            warmup_steps=burnin_steps)
    mcmc.run(use_reference)
    #mcmc.summary()
    return mcmc.get_samples()["theta"]


def compute_resultants(theta_samples):
    """
    Resultant length computation.

    theta_samples: MCMC samples of angles of the von Mises random walk

    returns: resultant length R
    """
    cos_theta = torch.cos(theta_samples)
    sin_theta = torch.sin(theta_samples)
    C = torch.mean(cos_theta, dim=1)
    S = torch.mean(sin_theta, dim=1)
    R = N * torch.sqrt(C**2 + S**2)
    return R


def plot_resultant_distribution(pk_R, ablation_R, prior_R):
    """
    Plot PDFs of prior, PK posterior and Ablation (posterior without reference).

    pk_R: resultant length samples of PK posterior
    ablation_R: resultant length samples of ablation (no reference distribution)
    prior_R: resultant length samples from unmodified VRW prior
    """
    # Set font size
    plt.rcParams.update({'font.size': 30})

    # x-values for plot
    R_vals = torch.linspace(0.01, max_R, 500)

    # Prior density - following Stephens
    prior_density = np.exp(stephens_logpdf(R_vals, N, kappa_prior).numpy())

    # Target density - scaled Beta
    target_density = np.exp(beta_logpdf_scaled(R_vals, alpha_beta, beta_beta, max_R).numpy())

    # Figure settings
    plt.figure(figsize=(15, 10))

    # Histogram of samples from resultant length R according to the VRW
    plt.hist(prior_R, bins=25, color='blue', density=True, alpha=0.4, label="Empirical prior")
    # Stephens prior (PDF from theory, Stephens, 1969)
    plt.plot(R_vals, prior_density, '--', color='black', label="Theoretical prior (Stephens)", lw=2)

    # PK update - histogram of MCMC samples
    plt.hist(pk_R, bins=25, density=True, color='red', alpha=0.4, label="Empirical posterior")
    # PK update - target density (scaled Beta)
    plt.plot(R_vals, target_density, color='black', label=f"Theoretical posterior (ScaledBeta)", lw=2)

    # Udate without reference (Ablation)
    plt.hist(ablation_R, bins=25, density=True, color='grey', alpha=0.4, label="Naive product")

    plt.legend()

    plt.xlabel("$d$")
    plt.ylabel("PDF")
    #plt.title("Resultant Distribution: Prior vs Posterior vs Target")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vrw_densities.png")
    #plt.show()


def run_ks_test(samples_R):
    """
    Perform Kolmogorov–Smirnov Test to test fit of PK update / Ablation update
    to target distribution (scaled Beta).

    samples_R: MCMC samples of resultant length

    returns: (D, p-value)
    """
    from scipy.stats import beta
    # Thin the MCMC samples
    samples_scaled = samples_R / max_R
    # Take every 5th sample for test, as MCMC samples are correlated
    D, p_value = ks_1samp(samples_scaled[::5], beta(alpha_beta, beta_beta).cdf)
    print(f"KS Test of r/N vs Beta({alpha_beta},{beta_beta}): D = {D:.6f}, p = {p_value:.6f}")
    return D, p_value


def sample_vrw_prior():
    """
    Sample trajectories from the unmodified von Mises random walk with defaut parameters. 
    """
    theta = dist.VonMises(0.0, kappa_prior).sample((num_samples, N))
    return theta


def angles_to_2D_coordinates(theta):
    """
    Calculate resultant lengths from random walk angles.

    theta: (n x N) array of angles, where n is number of walks
    """
    n = theta.shape[0]
    steps = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    walks = torch.cat([torch.zeros(n, 1, 2), torch.cumsum(steps, dim=1)], dim=1)
    return walks


def plot_walks(theta_pk_samples, theta_prior_samples):
    """
    Plot trajectories for unmodified von Mises random walk and PK update.

    theta_pk_samples: sampled angles of PK updated von Mises random walk
    theta_prior_samples: sampled angles of unmodified von Mises random walk
    """
    # 2D coordinates from PK update 
    walks_rrm = angles_to_2D_coordinates(theta_pk_samples)

    # 2D coordinates from unmodified von Mises random walk
    walks_prior = angles_to_2D_coordinates(theta_prior_samples)

    # Figure settings
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(15, 10))

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    for i in random.sample(range(0, num_samples), n_trajectories):
        wp = walks_prior[i].numpy()
        wr = walks_rrm[i].numpy()
        # Unmodified random walk in blue
        plt.plot(wp[:, 0], wp[:, 1], 'b', alpha=0.1)
        # Endpoint
        plt.plot(wp[-1, 0], wp[-1, 1], 'bo', alpha=1.0)
        # PK updated random walk in red
        plt.plot(wr[:, 0], wr[:, 1], 'r', alpha=0.1)
        # Endpoint
        plt.plot(wr[-1, 0], wr[-1, 1], 'ro', alpha=1.0)
        # Black dot at origin (0,0)
        plt.plot(wr[0, 0], wr[0, 1], 'ko', alpha=1.0)
    #plt.title("Random Walks: Prior (Blue) vs RRM (Red)")
    #plt.axis("equal")
    plt.xlim(-2, 6.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vrw_trajectories.png")
    #plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv)>1:
        # Set a different seed
        try:
            set_seed(int(sys.argv[1]))
        except:
            print("Usage: python vonmises_rw.py <int>")  
            sys.exit()
    else:
        set_seed(42)

    # We do 10 KS hypothesis tests
    # D = KS statistic, p = p-value
    pk_D=[]
    pk_p=[]
    ab_D=[]
    ab_p=[]

    for i in range(0, 10):
        # Angles from PK update
        pk_samples = run_inference(True)
        # Angles from Ablation (native product, no reference distribution)
        ablation_samples = run_inference(False)
        # Angles from unmodified VRW prior
        prior_samples = sample_vrw_prior() 

        # Calculate resultant lengths 
        #...from PK update
        pk_R = compute_resultants(pk_samples).numpy()
        #...from Ablation
        ablation_R = compute_resultants(ablation_samples).numpy()
        # ...from unmodified prior
        prior_R = compute_resultants(prior_samples).numpy()

        # Hypothesis test - do the samples fit the ScaledBeta target?
        print("PK Update: ", end="")
        D, p = run_ks_test(pk_R)
        pk_D.append(D)
        pk_p.append(p)
        print("Ablation: ", end="")
        D, p = run_ks_test(ablation_R)
        ab_D.append(D)
        ab_p.append(p)

    # KS Results
    print("PK update min D: \t", np.min(pk_D)) 
    print("PK update min p: \t", np.min(pk_p)) 
    print("Ablation min D: \t", np.min(ab_D)) 
    print("Alation min p: \t", np.min(ab_p)) 
    print("PK update median D \t: ", np.median(pk_D)) 
    print("PK update median p: \t", np.median(pk_p)) 
    print("Ablation median D: \t", np.median(ab_D)) 
    print("Ablation median p: \t", np.median(ab_p)) 
    print("PK update max D: \t", np.max(pk_D)) 
    print("PK update max p: \t", np.max(pk_p)) 
    print("Ablation max D: \t", np.max(ab_D)) 
    print("Ablation max p: \t", np.max(ab_p)) 

    # Plot PDFs and histograms of last MCMC run
    plot_resultant_distribution(pk_R, ablation_R, prior_R)

    # Plot trajectories of last MCMC run
    plot_walks(pk_samples, prior_samples)
