def compute(tau=None, T=None, N=None, D=None):
    # tau*T = 6*N*D
    lhs = 1
    rhs = 6
    n_none = 4
    if N is not None:
        rhs *= N * 10**6
        n_none -= 1
    if D is not None:
        rhs *= D * 10**9
        n_none -= 1
    if tau is not None:
        lhs *= tau * 10**12
        n_none -= 1
    if T is not None:
        lhs *= T * 24 * 3600
        n_none -= 1

    assert n_none == 1, "Must have exactly one of N, D, tau, T equal to None"

    if N is None:
        print(f"Model size: {lhs/rhs:.2e} weights")
    if D is None:
        print(f"Dataset size: {lhs/rhs:.2e} tokens")
    if tau is None:
        print(f"FLOPS requirement: {rhs/lhs:.2d}")
    if T is None:
        print(f"Training time: {rhs/lhs/(24*3600):.2f} days")
