import numpy as np
from scipy.integrate import quad
from scipy.stats import rv_discrete

def psibarra(beta, k, q, num_terms=100):
    """
    Computes the Psibarra value for a given integer k and parameter beta.

    Parameters:
    beta (float): A fixed parameter that controls the shape of the exponential function.
    k (int): The integer index for which to compute Psibarra.
    q (int): Determines the range of integration bounds.
    num_terms (int): The number of terms in the summation (-num_terms to +num_terms).

    Returns:
    float: The computed value of Psibarra for the given parameters.
    """
    def integrand(x, i, beta):
        return np.exp(-np.pi / beta**2 * (x - i)**2)

    total_sum = 0
    for i in range(-num_terms, num_terms + 1):  # Symmetric range for the summation
        lower_bound = (k - 0.5) / q
        upper_bound = (k + 0.5) / q
        integral, _ = quad(integrand, lower_bound, upper_bound, args=(i, beta))
        total_sum += integral

    return total_sum / beta

def create_empirical_distribution(beta, q, num_terms=100):
    """
    Creates a discrete empirical distribution based on possible values and their probabilities.

    Parameters:
    beta (float): A fixed parameter that controls the Psibarra function.
    q (int): The number of possible discrete values (0 to q-1).
    num_terms (int): The number of terms in the summation (-num_terms to +num_terms).

    Returns:
    rv_discrete: A discrete random variable with the specified probabilities.
    """
    possible_values = np.arange(q)  # Possible values: 0, 1, ..., q-1
    probabilities = np.array([psibarra(beta, k, q, num_terms) for k in possible_values])
    probabilities /= probabilities.sum()  # Normalize probabilities to ensure they sum to 1
    return rv_discrete(values=(possible_values, probabilities))

def generate_samples(beta, q, num_samples, num_terms=100):
    """
    Generates samples from the empirical discrete distribution.

    Parameters:
    beta (float): A fixed parameter that controls the Psibarra function.
    q (int): The number of possible discrete values (0 to q-1).
    num_samples (int): The number of samples to generate.
    num_terms (int): The number of terms in the summation (-num_terms to +num_terms).

    Returns:
    numpy.ndarray: An array of generated samples following the empirical distribution.
    """
    discrete_distribution = create_empirical_distribution(beta, q, num_terms)
    return discrete_distribution.rvs(size=num_samples)

# Example usage
if __name__ == "__main__":
    beta = 0.5  # Fixed beta parameter
    q = 10      # Number of possible discrete values
    num_samples = 1000  # Number of samples to generate

    samples = generate_samples(beta, q, num_samples)
    print(samples)
