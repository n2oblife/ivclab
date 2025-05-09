def first_order_predictor(r0, r1):
    """
    Computes the optimal coefficient a1 for a first-order linear predictor
    using the Wiener-Hopf equation.

    Parameters:
        r0 (float): Autocorrelation at lag 0 (r(0))
        r1 (float): Autocorrelation at lag 1 (r(1))

    Returns:
        a1 (float): Optimal predictor coefficient
    """
    if r0 == 0:
        raise ValueError("Autocorrelation at lag 0 (r0) must be non-zero.")
    return r1 / r0


a1 = first_order_predictor(r0=0.3040, r1=0.3033)
print(f"Optimal predictor coefficient a1: {a1:.4f}")
