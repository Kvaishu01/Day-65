import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Objective function (example: minimize f(x) = x^2)
def objective_function(x):
    return x ** 2 + 10 * np.sin(x)

# Simulated Annealing Algorithm
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # Start with a random solution
    current = bounds[0] + np.random.rand() * (bounds[1] - bounds[0])
    current_eval = objective(current)
    best, best_eval = current, current_eval
    scores = []

    for i in range(n_iterations):
        # Take a step
        candidate = current + np.random.randn() * step_size
        # Evaluate candidate
        candidate_eval = objective(candidate)

        # Check for improvement
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval

        # Calculate change and acceptance probability
        diff = candidate_eval - current_eval
        t = temp / float(i + 1)
        metropolis = np.exp(-diff / t)

        # Accept the candidate if it’s better or probabilistically
        if diff < 0 or np.random.rand() < metropolis:
            current, current_eval = candidate, candidate_eval

        scores.append(best_eval)

    return [best, best_eval, scores]

# Streamlit UI
st.title("🔥 Simulated Annealing Optimization")
st.write("An interactive demo of Simulated Annealing — inspired by the physical process of heating and slowly cooling metal to reduce defects.")

# Sidebar parameters
n_iterations = st.slider("Number of Iterations", 100, 5000, 1000)
step_size = st.slider("Step Size", 0.01, 2.0, 0.1)
temp = st.slider("Initial Temperature", 1.0, 100.0, 10.0)
bounds = [-10.0, 10.0]

if st.button("Run Optimization"):
    best, score, scores = simulated_annealing(objective_function, bounds, n_iterations, step_size, temp)

    st.success(f"✅ Best Solution Found: x = {best:.4f}, f(x) = {score:.4f}")

    # Plot convergence
    fig, ax = plt.subplots()
    ax.plot(scores)
    ax.set_title("Simulated Annealing Convergence Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Objective Value")
    st.pyplot(fig)

    # Plot function landscape
    x = np.linspace(bounds[0], bounds[1], 1000)
    y = objective_function(x)
    fig2, ax2 = plt.subplots()
    ax2.plot(x, y, label="Objective Function")
    ax2.scatter(best, score, color='red', label="Best Solution")
    ax2.set_title("Objective Function Landscape")
    ax2.legend()
    st.pyplot(fig2)
