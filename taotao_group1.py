"""

This script shows:
1. how the gravity anomaly changes with depth
2. how the condition number of the Jacobian changes with depth
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# constants
G = 6.674e-11
MPS2_TO_MGAL = 1e5


def gravity_anomaly(x, theta):
    """
    Gravity anomaly from a buried point mass.

    theta = [m, s, h]
    m : mass
    s : horizontal position
    h : depth
    """
    m, s, h = theta
    r2 = (x - s) ** 2 + h ** 2
    mu = G * m * h / (r2 ** 1.5)
    return mu * MPS2_TO_MGAL


def jacobian(x, theta):
    """
    Jacobian of mu with respect to [m, s, h]
    """
    m, s, h = theta
    r2 = (x - s) ** 2 + h ** 2

    J = np.zeros((len(x), 3))

    # dmu/dm
    J[:, 0] = G * h / (r2 ** 1.5) * MPS2_TO_MGAL

    # dmu/ds
    J[:, 1] = 3 * G * m * h * (x - s) / (r2 ** 2.5) * MPS2_TO_MGAL

    # dmu/dh
    J[:, 2] = G * m * ((x - s) ** 2 - 2 * h ** 2) / (r2 ** 2.5) * MPS2_TO_MGAL

    return J


def plot_forward_curves():
    x = np.linspace(-10, 10, 400)
    m = 1e6
    s = 0
    depths = [1, 3, 5, 10]

    plt.figure(figsize=(8, 5))

    for h in depths:
        theta = [m, s, h]
        mu = gravity_anomaly(x, theta)
        plt.plot(x, mu, label=f"h = {h}")

    plt.xlabel("x")
    plt.ylabel("gravity anomaly (mGal)")
    plt.title("Gravity anomaly for different depths")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure1_forward_curves.png", dpi=150)
    plt.show()


def plot_condition_number():
    x = np.linspace(-10, 10, 80)
    m = 1e6
    s = 0
    depths = np.linspace(1, 15, 30)

    cond_values = []

    for h in depths:
        theta = [m, s, h]
        J = jacobian(x, theta)
        _, S, _ = svd(J, full_matrices=False)
        cond = S[0] / S[-1]
        cond_values.append(cond)

    plt.figure(figsize=(8, 5))
    plt.semilogy(depths, cond_values, "o-", linewidth=2)
    plt.xlabel("depth h")
    plt.ylabel("condition number")
    plt.title("Condition number of the Jacobian vs depth")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure2_condition_number.png", dpi=150)
    plt.show()

    print("Depths:", depths)
    print("Condition numbers:", cond_values)


if __name__ == "__main__":
    print("Running gravity inversion demo...")

    plot_forward_curves()
    plot_condition_number()

    print("Done.")