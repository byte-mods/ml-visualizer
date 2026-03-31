"""
Probability Distributions Visualizations
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List, Dict, Any
import streamlit as st
import pandas as pd
from scipy import stats


def create_distribution_plot(
    distribution_type: str,
    params: Dict[str, float],
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 1000,
    show_pdf: bool = True,
    show_cdf: bool = False,
    show_samples: bool = False,
    n_samples: int = 1000
) -> go.Figure:
    """
    Create a plot for a probability distribution

    Args:
        distribution_type: Type of distribution
        params: Parameters for the distribution
        x_range: Range of x values to plot
        n_points: Number of points for PDF/CDF
        show_pdf: Whether to show probability density function
        show_cdf: Whether to show cumulative distribution function
        show_samples: Whether to show random samples
        n_samples: Number of random samples to generate

    Returns:
        plotly.graph_objects.Figure
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    fig = go.Figure()

    # Get distribution object
    dist = get_distribution(distribution_type, params)

    # Plot PDF
    if show_pdf and hasattr(dist, 'pdf'):
        pdf = dist.pdf(x)
        fig.add_trace(go.Scatter(
            x=x, y=pdf,
            mode='lines',
            name='PDF',
            line=dict(color='blue', width=2),
            fill='tozeroy' if not show_cdf else None,
            fillcolor='rgba(0, 0, 255, 0.1)'
        ))

    # Plot CDF
    if show_cdf and hasattr(dist, 'cdf'):
        cdf = dist.cdf(x)
        fig.add_trace(go.Scatter(
            x=x, y=cdf,
            mode='lines',
            name='CDF',
            line=dict(color='red', width=2, dash='dash'),
            yaxis='y2'
        ))

        # Add secondary y-axis for CDF
        fig.update_layout(
            yaxis2=dict(
                title="CDF",
                overlaying='y',
                side='right',
                range=[0, 1]
            )
        )

    # Plot random samples
    if show_samples and hasattr(dist, 'rvs'):
        samples = dist.rvs(size=n_samples)
        fig.add_trace(go.Histogram(
            x=samples,
            name='Samples',
            opacity=0.5,
            nbinsx=30,
            marker_color='green',
            yaxis='y'
        ))

    # Update layout
    title = f"{distribution_type} Distribution"
    if params:
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        title += f" ({param_str})"

    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="Probability Density",
        hovermode='x unified',
        showlegend=True
    )

    return fig


def get_distribution(distribution_type: str, params: Dict[str, float]):
    """
    Get a scipy.stats distribution object

    Args:
        distribution_type: Type of distribution
        params: Parameters for the distribution

    Returns:
        scipy.stats distribution object
    """
    if distribution_type == "Normal (Gaussian)":
        return stats.norm(loc=params.get('mu', 0), scale=params.get('sigma', 1))
    elif distribution_type == "Uniform":
        return stats.uniform(loc=params.get('a', 0), scale=params.get('b', 1) - params.get('a', 0))
    elif distribution_type == "Binomial":
        return stats.binom(n=int(params.get('n', 10)), p=params.get('p', 0.5))
    elif distribution_type == "Exponential":
        return stats.expon(scale=1/params.get('lambda', 1))
    elif distribution_type == "Beta":
        return stats.beta(a=params.get('alpha', 2), b=params.get('beta', 2))
    elif distribution_type == "Gamma":
        return stats.gamma(a=params.get('shape', 2), scale=1/params.get('rate', 1))
    elif distribution_type == "Student's t":
        return stats.t(df=params.get('df', 10))
    elif distribution_type == "Chi-squared":
        return stats.chi2(df=params.get('df', 5))
    elif distribution_type == "Poisson":
        return stats.poisson(mu=params.get('mu', 5))
    else:
        return stats.norm()  # Default to standard normal


def create_multiple_distributions_comparison(
    distributions: List[Tuple[str, Dict[str, float], str]],
    x_range: Tuple[float, float] = (-5, 5),
    n_points: int = 1000,
    plot_type: str = 'pdf'  # 'pdf', 'cdf', or 'both'
) -> go.Figure:
    """
    Compare multiple distributions on the same plot

    Args:
        distributions: List of (distribution_type, params, label) tuples
        x_range: Range of x values
        n_points: Number of points
        plot_type: Type of plot ('pdf', 'cdf', or 'both')

    Returns:
        plotly.graph_objects.Figure
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    fig = go.Figure()

    for dist_type, params, label in distributions:
        dist = get_distribution(dist_type, params)

        if plot_type in ['pdf', 'both'] and hasattr(dist, 'pdf'):
            pdf = dist.pdf(x)
            fig.add_trace(go.Scatter(
                x=x, y=pdf,
                mode='lines',
                name=f"{label} PDF",
                line=dict(width=2)
            ))

        if plot_type in ['cdf', 'both'] and hasattr(dist, 'cdf'):
            cdf = dist.cdf(x)
            fig.add_trace(go.Scatter(
                x=x, y=cdf,
                mode='lines',
                name=f"{label} CDF",
                line=dict(width=2, dash='dash')
            ))

    fig.update_layout(
        title="Distribution Comparison",
        xaxis_title="x",
        yaxis_title="Probability Density / Cumulative Probability",
        hovermode='x unified'
    )

    return fig


def create_central_limit_theorem_demo(
    population_distribution: str,
    population_params: Dict[str, float],
    sample_size: int = 30,
    n_samples: int = 1000,
    n_bins: int = 30
) -> go.Figure:
    """
    Demonstrate the Central Limit Theorem

    Args:
        population_distribution: Distribution of population
        population_params: Parameters for population distribution
        sample_size: Size of each sample
        n_samples: Number of samples to draw
        n_bins: Number of bins for histogram

    Returns:
        plotly.graph_objects.Figure with subplots
    """
    # Create population distribution
    pop_dist = get_distribution(population_distribution, population_params)

    # Draw samples and compute means
    sample_means = []
    for _ in range(n_samples):
        sample = pop_dist.rvs(size=sample_size)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # Create subplots
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Population Distribution",
            f"Sample Means (n={sample_size})",
            "QQ Plot (Normality Check)",
            "Statistics"
        ),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}]
        ]
    )

    # 1. Population distribution
    x_pop = np.linspace(pop_dist.ppf(0.001), pop_dist.ppf(0.999), 1000)
    pdf_pop = pop_dist.pdf(x_pop)
    fig.add_trace(
        go.Scatter(x=x_pop, y=pdf_pop, mode='lines', name='Population PDF'),
        row=1, col=1
    )

    # 2. Distribution of sample means
    fig.add_trace(
        go.Histogram(x=sample_means, nbinsx=n_bins, name='Sample Means',
                    histnorm='probability density'),
        row=1, col=2
    )

    # Add normal curve for comparison
    mean_of_means = np.mean(sample_means)
    std_of_means = np.std(sample_means)
    x_norm = np.linspace(mean_of_means - 4*std_of_means,
                        mean_of_means + 4*std_of_means, 1000)
    pdf_norm = stats.norm.pdf(x_norm, mean_of_means, std_of_means)
    fig.add_trace(
        go.Scatter(x=x_norm, y=pdf_norm, mode='lines', name='Normal Approx',
                  line=dict(color='red', dash='dash')),
        row=1, col=2
    )

    # 3. QQ plot for normality check
    from scipy import stats as sp_stats
    (osm, osr), (slope, intercept, r) = sp_stats.probplot(sample_means, dist="norm")
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode='markers', name='Sample Quantiles'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=osm, y=slope*osm + intercept, mode='lines',
                  name=f'Theoretical (r={r:.3f})', line=dict(color='red')),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)

    # 4. Statistics table
    stats_table = go.Table(
        header=dict(
            values=['Statistic', 'Value'],
            fill_color='lightblue',
            align='left'
        ),
        cells=dict(
            values=[
                ['Population Mean', 'Population Std', 'Sample Mean', 'Std of Means',
                 'Skewness', 'Kurtosis', 'Normality (p-value)'],
                [f'{pop_dist.mean():.3f}', f'{pop_dist.std():.3f}',
                 f'{mean_of_means:.3f}', f'{std_of_means:.3f}',
                 f'{sp_stats.skew(sample_means):.3f}',
                 f'{sp_stats.kurtosis(sample_means):.3f}',
                 f'{sp_stats.normaltest(sample_means)[1]:.3f}']
            ],
            align='left'
        )
    )
    fig.add_trace(stats_table, row=2, col=2)

    fig.update_layout(height=800, showlegend=False, title_text="Central Limit Theorem Demonstration")

    return fig


def show_distributions_ui():
    """Streamlit UI for distributions visualization"""
    st.markdown("## 🔔 Probability Distributions")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Controls")

        distribution_type = st.selectbox(
            "Distribution Type",
            [
                "Normal (Gaussian)",
                "Uniform",
                "Binomial",
                "Exponential",
                "Beta",
                "Gamma",
                "Student's t",
                "Chi-squared",
                "Poisson"
            ]
        )

        # Distribution-specific parameters
        params = {}
        if distribution_type == "Normal (Gaussian)":
            params['mu'] = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
            params['sigma'] = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)

        elif distribution_type == "Uniform":
            params['a'] = st.slider("Lower bound (a)", -5.0, 5.0, 0.0, 0.1)
            params['b'] = st.slider("Upper bound (b)", -5.0, 5.0, 1.0, 0.1)
            if params['b'] <= params['a']:
                params['b'] = params['a'] + 0.1
                st.warning("Upper bound must be greater than lower bound")

        elif distribution_type == "Binomial":
            params['n'] = st.slider("Number of trials (n)", 1, 50, 10, 1)
            params['p'] = st.slider("Success probability (p)", 0.0, 1.0, 0.5, 0.01)

        elif distribution_type == "Exponential":
            params['lambda'] = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)

        elif distribution_type == "Beta":
            params['alpha'] = st.slider("Alpha (α)", 0.1, 10.0, 2.0, 0.1)
            params['beta'] = st.slider("Beta (β)", 0.1, 10.0, 2.0, 0.1)

        elif distribution_type == "Gamma":
            params['shape'] = st.slider("Shape (k)", 0.1, 10.0, 2.0, 0.1)
            params['rate'] = st.slider("Rate (θ)", 0.1, 5.0, 1.0, 0.1)

        elif distribution_type == "Student's t":
            params['df'] = st.slider("Degrees of freedom (ν)", 1, 50, 10, 1)

        elif distribution_type == "Chi-squared":
            params['df'] = st.slider("Degrees of freedom (k)", 1, 50, 5, 1)

        elif distribution_type == "Poisson":
            params['mu'] = st.slider("Mean (μ)", 0.1, 20.0, 5.0, 0.1)

        # Plot options
        st.markdown("### Plot Options")
        show_pdf = st.checkbox("Show PDF", True)
        show_cdf = st.checkbox("Show CDF", False)
        show_samples = st.checkbox("Show Random Samples", False)

        if show_samples:
            n_samples = st.slider("Number of samples", 10, 10000, 1000, 10)

        x_min = st.number_input("X min", -20.0, 20.0, -5.0, 0.5)
        x_max = st.number_input("X max", -20.0, 20.0, 5.0, 0.5)
        if x_max <= x_min:
            x_max = x_min + 1.0
            st.warning("X max must be greater than X min")

        # Central Limit Theorem demo
        st.markdown("---")
        show_clt = st.checkbox("Show Central Limit Theorem Demo", False)
        if show_clt:
            clt_sample_size = st.slider("Sample size for CLT", 2, 100, 30, 1)
            clt_n_samples = st.slider("Number of samples for CLT", 100, 10000, 1000, 100)

    with col1:
        st.markdown("### Distribution Visualization")

        # Create the main distribution plot
        fig = create_distribution_plot(
            distribution_type=distribution_type,
            params=params,
            x_range=(x_min, x_max),
            show_pdf=show_pdf,
            show_cdf=show_cdf,
            show_samples=show_samples if 'show_samples' in locals() else False,
            n_samples=n_samples if 'n_samples' in locals() else 1000
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show CLT demo if requested
        if 'show_clt' in locals() and show_clt:
            st.markdown("### Central Limit Theorem Demonstration")
            clt_fig = create_central_limit_theorem_demo(
                population_distribution=distribution_type,
                population_params=params,
                sample_size=clt_sample_size,
                n_samples=clt_n_samples
            )
            st.plotly_chart(clt_fig, use_container_width=True)

    # Educational content
    st.markdown("---")
    st.markdown("## 📚 Chapter 2: Probability Distributions - Tutorial")

    st.markdown("""
    <div style="background-color: #e8f4f8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
    <b>Learning Objectives:</b> Understand probability distributions as the language of uncertainty in ML.
    Master key distributions, their mathematical formulas, and how they appear in ML algorithms.
    </div>
    """, unsafe_allow_html=True)

    # Mathematical Foundations
    st.markdown("### 📖 Mathematical Foundations")

    with st.expander("📐 Core Probability Formulas"):
        st.markdown("""
        **Probability Density Function (PDF):**
        For continuous random variable X, the PDF f(x) satisfies:

        ```
        P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx

        Properties:
        - f(x) ≥ 0 for all x
        - ∫_{-∞}^{+∞} f(x)dx = 1
        ```

        **Cumulative Distribution Function (CDF):**
        ```
        F(x) = P(X ≤ x) = ∫_{-∞}^{x} f(t)dt

        Properties:
        - 0 ≤ F(x) ≤ 1
        - P(X > x) = 1 - F(x)
        - P(a < X ≤ b) = F(b) - F(a)
        ```

        **Expected Value (Mean):**
        ```
        E[X] = ∫_{-∞}^{+∞} x · f(x)dx      (continuous)
        E[X] = Σ x · P(X = x)               (discrete)

        Properties:
        - E[aX + b] = aE[X] + b
        - E[X + Y] = E[X] + E[Y]
        ```

        **Variance:**
        ```
        Var(X) = E[(X - E[X])²] = E[X²] - E[X]²

        Properties:
        - Var(aX + b) = a²Var(X)
        - Var(X + Y) = Var(X) + Var(Y)  (if independent)
        ```

        **Standard Deviation:** σ = √Var(X)
        """)

    with st.expander("📊 Distribution-Specific Formulas"):
        st.markdown(f"""
        **Normal (Gaussian) Distribution:** N(μ, σ²)
        ```
        PDF: f(x) = (1/√(2πσ²)) · exp(-(x-μ)²/2σ²)

        CDF: F(x) = (1/2)[1 + erf((x-μ)/σ√2)]

        Special case: Standard Normal N(0,1):
        - μ = 0, σ = 1
        - f(x) = (1/√(2π)) · exp(-x²/2)
        ```

        **Uniform Distribution:** U(a, b)
        ```
        PDF: f(x) = 1/(b-a) for a ≤ x ≤ b, else 0

        CDF: F(x) = 0 for x < a
            = (x-a)/(b-a) for a ≤ x ≤ b
            = 1 for x > b

        Mean: E[X] = (a+b)/2
        Variance: Var(X) = (b-a)²/12
        ```

        **Exponential Distribution:** Exp(λ)
        ```
        PDF: f(x) = λe^{-λx} for x ≥ 0

        CDF: F(x) = 1 - e^{-λx} for x ≥ 0

        Mean: E[X] = 1/λ
        Variance: Var(X) = 1/λ²
        Memoryless property: P(X > s + t | X > s) = P(X > t)
        ```

        **Beta Distribution:** Beta(α, β)
        ```
        PDF: f(x) = x^{α-1}(1-x)^{β-1} / B(α,β)
             where B(α,β) = Γ(α)Γ(β)/Γ(α+β)

        Mean: E[X] = α/(α+β)
        Variance: Var(X) = αβ/[(α+β)²(α+β+1)]
        ```
        """)

    with st.expander("🔢 Bayesian Statistics Formulas"):
        st.markdown("""
        **Bayes' Theorem:**
        ```
        P(θ|D) = P(D|θ) · P(θ) / P(D)

        where:
        - P(θ|D) = Posterior (what we believe after seeing data)
        - P(D|θ) = Likelihood (how likely is data given hypothesis)
        - P(θ)   = Prior (our belief before seeing data)
        - P(D)   = Evidence (normalizing constant)
        ```

        **Maximum Likelihood Estimation (MLE):**
        ```
        θ_MLE = argmax_θ P(D|θ)
              = argmax_θ log P(D|θ)

        For i.i.d. data: log P(D|θ) = Σ log P(x_i|θ)
        ```

        **Maximum A Posteriori (MAP):**
        ```
        θ_MAP = argmax_θ P(θ|D)
              = argmax_θ P(D|θ) · P(θ)
        ```

        **Conjugate Priors (common in ML):**
        | Likelihood     | Prior      | Posterior |
        |----------------|------------|-----------|
        | Bernoulli      | Beta       | Beta      |
        | Binomial       | Beta       | Beta      |
        | Poisson        | Gamma      | Gamma     |
        | Gaussian (σ²)  | Inverse-γ  | Inverse-γ |
        | Gaussian (μ)   | Gaussian   | Gaussian  |
        """)

    # ML Applications
    st.markdown("### 🛠️ Machine Learning Applications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **1. Normal Distribution in ML**

        - **Weight Initialization:** `W ~ N(0, 0.01)`
        - **Activation Distribution:** Pre-activation z ~ N(0,1) via batch norm
        - **Gaussian Naive Bayes:** Class conditional densities are Gaussian
        - **Gaussian Processes:** Prior over functions is GP with Gaussian marginals

        **2. Exponential Distribution**

        - **Dropout survival times:** Time until neuron drops out
        - **Reinforcement learning:** Exponential reward discounting
        - **Survival analysis in ML:** Predicting time-to-event

        **3. Beta Distribution**

        - **Bayesian A/B testing:** Model conversion rates
        - **Thompson sampling:** Sample from Beta for exploration
        - **Uncertainty in probabilities:** Bernoulli parameter π ~ Beta(α,β)
        """)

    with col2:
        st.markdown("""
        **4. Sampling in ML**

        - **Monte Carlo Estimation:** E[f(X)] ≈ (1/n)Σf(x_i)
        - **Variance Reduction:** Importance sampling, antithetic variates
        - **Markov Chain Monte Carlo (MCMC):** Gibbs sampling, Metropolis-Hastings

        **5. Loss Functions as Distributions**

        - **Cross-Entropy:** Measures distance between true p and predicted q
        ```
        H(p,q) = -Σ p(x) log q(x)
        ```

        - **Gaussian NLL:** Negative log-likelihood of Gaussian
        ```
        NLL = -log N(y|μ(x), σ²)
        ```

        **6. Central Limit Theorem in Practice**
        ```
        X̄ ~ N(μ, σ²/n) as n → ∞

        Confidence Interval: μ ∈ [X̄ ± z_{α/2} · σ/√n]
        ```
        """)

    # Theorems
    st.markdown("### 📜 Key Theorems")

    with st.expander("📈 Central Limit Theorem (CLT)"):
        st.markdown("""
        **Statement:** If X₁, X₂, ..., Xₙ are i.i.d. with mean μ and variance σ²,
        then as n → ∞:

        ```
        Z_n = (X̄_n - μ) / (σ/√n) → N(0,1) in distribution
        ```

        **Why it matters in ML:**
        - Justifies using Gaussian assumptions for large samples
        - Explains why batch norms work (sum of many random variables → Gaussian)
        - Enables statistical inference on model parameters
        - Confidence intervals for model performance metrics
        """)

    with st.expander("📉 Law of Large Numbers (LLN)"):
        st.markdown("""
        **Weak LLN:** If X₁, X₂, ..., Xₙ are i.i.d. with E[|X_i|] < ∞:
        ```
        X̄_n → E[X] in probability as n → ∞
        ```

        **Why it matters in ML:**
        - Sample means converge to population means
        - Empirical risk minimization converges to expected risk
        - Gradient estimation via Monte Carlo is consistent
        - Stochastic gradient descent (SGD) converges
        """)

    # Key Takeaways
    st.markdown("### 🎯 Key Takeaways")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
    1. <b>Distributions describe uncertainty</b> - Every ML model makes probabilistic predictions

    2. <b>Normal is everywhere</b> - CLT ensures sum of random variables tends to Gaussian

    3. <b>Conjugate priors simplify Bayesian inference</b> - Closed-form posterior updates

    4. <b>Loss functions are negative log-likelihoods</b> - Cross-entropy, MSE all derive from likelihood principle

    5. <b>Sampling enables estimation</b> - MC methods approximate expectations we can't compute analytically
    </div>
    """, unsafe_allow_html=True)

    # Quiz
    with st.expander("📝 Quick Quiz: Test Your Understanding"):
        st.markdown("""
        **Q1:** If X ~ N(0,1) and Y ~ N(0,1) are independent, what is X + Y?
        - (a) N(0, 1)
        - (b) N(0, 2)
        - (c) Cannot determine

        **Q2:** What conjugate prior would you use for a Bernoulli likelihood?
        - (a) Gaussian
        - (b) Beta
        - (c) Exponential
        - (d) Uniform

        **Q3:** The CLT justifies using ___ assumptions for large samples in ML.
        - (a) Uniform
        - (b) Exponential
        - (c) Gaussian
        - (d) Poisson
        """)

    # Distribution comparison
    st.markdown("---")
    st.markdown("### Distribution Comparison")

    if st.checkbox("Compare multiple distributions"):
        n_comparisons = st.slider("Number of distributions to compare", 2, 5, 3, 1)

        distributions = []
        colors = px.colors.qualitative.Set1

        for i in range(n_comparisons):
            st.markdown(f"#### Distribution {i+1}")
            col_a, col_b = st.columns(2)
            with col_a:
                dist_type = st.selectbox(
                    f"Type {i+1}",
                    ["Normal (Gaussian)", "Uniform", "Exponential", "Beta", "Gamma"],
                    key=f"compare_type_{i}"
                )
            with col_b:
                # Simplified parameters for comparison
                if dist_type == "Normal (Gaussian)":
                    mu = st.slider(f"μ{i+1}", -3.0, 3.0, float(i), 0.1, key=f"mu_{i}")
                    sigma = st.slider(f"σ{i+1}", 0.1, 3.0, 1.0, 0.1, key=f"sigma_{i}")
                    params = {'mu': mu, 'sigma': sigma}
                elif dist_type == "Uniform":
                    a = st.slider(f"a{i+1}", -3.0, 3.0, float(i-1), 0.1, key=f"a_{i}")
                    b = st.slider(f"b{i+1}", -3.0, 3.0, float(i+1), 0.1, key=f"b_{i}")
                    params = {'a': a, 'b': b}
                elif dist_type == "Exponential":
                    lam = st.slider(f"λ{i+1}", 0.1, 3.0, 1.0, 0.1, key=f"lambda_{i}")
                    params = {'lambda': lam}
                elif dist_type == "Beta":
                    alpha = st.slider(f"α{i+1}", 0.1, 5.0, 2.0, 0.1, key=f"alpha_{i}")
                    beta = st.slider(f"β{i+1}", 0.1, 5.0, 2.0, 0.1, key=f"beta_{i}")
                    params = {'alpha': alpha, 'beta': beta}
                else:  # Gamma
                    shape = st.slider(f"k{i+1}", 0.1, 5.0, 2.0, 0.1, key=f"shape_{i}")
                    rate = st.slider(f"θ{i+1}", 0.1, 3.0, 1.0, 0.1, key=f"rate_{i}")
                    params = {'shape': shape, 'rate': rate}

            distributions.append((dist_type, params, f"Dist {i+1}"))

        if distributions:
            compare_fig = create_multiple_distributions_comparison(
                distributions,
                x_range=(x_min, x_max),
                plot_type='pdf'
            )
            st.plotly_chart(compare_fig, use_container_width=True)