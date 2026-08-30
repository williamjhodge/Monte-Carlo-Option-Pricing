
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.stats import norm

# %%

S0 = 100       # current stock price
r = 0.05       # interest rate (5% per year)
sigma = 0.2    # volatility (20% per year)
T = 1          # 1 year
K = 100       # strike price

num_simulations = 100000
Z = np.random.normal(size=num_simulations)
ST_ = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
plt.hist(ST_, bins=100)
plt.xlabel("Stock price after 1 year (£)")
plt.ylabel("Frequency")
plt.title("Simulated Stock Prices")
plt.show()

#Monte Carlo simulation for option pricing
def monte_carlo_option_price(S0, K, r, sigma, T, num_simulations=100000):
    Z = np.random.normal(size=num_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    return price

mc_price = monte_carlo_option_price(S0, K, r, sigma, T)
print(f"Monte Carlo estimated option price: £{mc_price:.4f}")

#Black-Scholes formula for European call option

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    C = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return C

bs_price = black_scholes_call(S0, K, r, sigma, T)
print(f"Black-Scholes price: £{bs_price:.4f}")



#investigate the effect of volatility on option price

sigmas = np.linspace(0.0001, 1, 1000)
plt.plot(sigmas, [np.exp(-r*T) * np.mean(np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z) - K, 0)) for sigma in sigmas])
plt.xlabel("Volatility")
plt.ylabel("Option Price")
plt.title("Effect of Volatility on Option Price")
plt.show()
# Option price increases with volatility.
# Over this range, the relationship appears approximately linear,
# although it is not exactly linear.

#investigate the effect of interest rate on option price
interest_rates = np.linspace(0, 0.5, 1000)
plt.plot(interest_rates, [np.exp(-rate*T) * np.mean(np.maximum(S0 * np.exp((rate - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z) - K, 0)) for rate in interest_rates])
plt.xlabel("Interest Rate")
plt.ylabel("Option Price")
plt.title("Effect of Interest Rate on Option Price")
plt.show()
#curves upwards; as interest rate increases, the option price increases as well.

#investigate the effect of strike price on option price
strike_prices = np.linspace(25, 200, 1000)
plt.plot(strike_prices, [np.exp(-r*T) * np.mean(np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z) - K, 0)) for K in strike_prices])
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.title("Effect of Strike Price on Option Price")
plt.show()
# As strike price increases, the call option price decreases.

#investigate the effect of expiry time on option price
expiry_times = np.linspace(0.1, 2, 1000)
plt.plot(expiry_times, [np.exp(-r*T) * np.mean(np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z) - K, 0)) for T in expiry_times])
plt.xlabel("Expiry Time")
plt.ylabel("Option Price")
plt.title("Effect of Expiry Time on Option Price")
plt.show()
# Option price generally increases as time to expiry increases.


#%%

#Investigate the convergence of Monte Carlo simulation to Black-Scholes price

error = mc_price - bs_price
absolute_error = abs(error)
percentage_error = absolute_error / bs_price * 100

print(f"Monte Carlo: £{mc_price:.4f}")
print(f"Black-Scholes: £{bs_price:.4f}")
print(f"Absolute error: £{absolute_error:.4f}")
print(f"Percentage error: {percentage_error:.2f}%")

#%%
simulation_numbers = np.logspace(1, 6, 100).astype(int)

num_trials = 30

rmse = []
mc_prices = []  

np.random.seed(42)

for n in simulation_numbers:
    errors_for_n = []
    prices_for_n = []
    for trial in range(num_trials):

        mc_price = monte_carlo_option_price(
            S0, K, r, sigma, T, n
        )

        errors_for_n.append(mc_price - bs_price)
        prices_for_n.append(mc_price)

    rmse_n = np.sqrt(np.mean(np.array(errors_for_n)**2))
    rmse.append(rmse_n)
    mc_prices.append(np.mean(prices_for_n))  

rmse = np.array(rmse)
mc_prices = np.array(mc_prices)  


plt.plot(simulation_numbers, mc_prices, label="Monte Carlo Price")
plt.axhline(bs_price, color="red", linestyle="--", label="Black-Scholes Price")
plt.xscale("log")
plt.xlabel("Number of Simulations")
plt.ylabel("Option Price (£)")
plt.title("Convergence of Monte Carlo Option Price to Black-Scholes Price")
plt.legend()
plt.show()


errors = np.abs(mc_prices - bs_price)

plt.plot(simulation_numbers, errors, label="Monte Carlo Error")

plt.plot(
    simulation_numbers,
    errors[0] * np.sqrt(simulation_numbers[0] / simulation_numbers),
    label="1/sqrt(N)",
    linestyle="--"
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of Simulations")
plt.ylabel("Absolute Error (£)")
plt.title("Monte Carlo Convergence")
plt.legend()
plt.show()

plt.plot(simulation_numbers, rmse, label="Monte Carlo RMSE")

plt.plot(
    simulation_numbers,
    rmse[0] * np.sqrt(simulation_numbers[0] / simulation_numbers),
    linestyle="--",
    label="1/sqrt(N)"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Number of Simulations")
plt.ylabel("RMSE (£)")
plt.title("Monte Carlo Error Convergence")
plt.legend()
plt.show()

log_N = np.log(simulation_numbers)
log_rmse = np.log(rmse)

slope, intercept = np.polyfit(log_N, log_rmse, 1)

print(f"Estimated convergence rate: {slope:.3f}") #expected to be around -0.5 for Monte Carlo methods from the central limit theorem.

# %%

#Slider graph

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.subplots_adjust(bottom=0.35)

x = ['Monte Carlo Price','Black-Scholes Price']
y = [mc_price, bs_price]
bars = ax1.bar(x, y, edgecolor="black", color=['red', 'yellow'])
ax1.set_ylim(0, 50)

x2 = ['Absolute Error','Percentage Error']
y2 = [absolute_error, percentage_error]
bars2 = ax2.bar(x2, y2, edgecolor="black", color=['lightblue', 'lightgreen'])
ax2.set_ylim(0, 2)

axr = plt.axes([0.25, 0.2, 0.65, 0.03])
axsigma = plt.axes([0.25, 0.15, 0.65, 0.03])
axK = plt.axes([0.25, 0.1, 0.65, 0.03])
axT = plt.axes([0.25, 0.05, 0.65, 0.03])

A = Slider(axr, 'Interest Rate', 0.0, 1.0, valinit=0.05)
B = Slider(axsigma, 'Volatility', 0.0, 1.0, valinit=0.5)
C = Slider(axK, 'Strike Price', 0.0, 200, valinit=100)
D = Slider(axT, 'Expiry Time', 0.0, 5, valinit=1)

def update(val):
    r = A.val
    sigma = B.val
    K = C.val
    T = D.val
    y = [np.exp(-r*T) * np.mean(np.maximum(S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z) - K, 0)), black_scholes_call(S0, K, r, sigma, T)]
    y2 = [abs(y[0] - y[1]), abs(y[0] - y[1]) / y[1] * 100]

    for i in range(len(bars)):
        bars[i].set_height(y[i])

    for i in range(len(bars2)):
        bars2[i].set_height(y2[i])

    fig.canvas.draw_idle()


A.on_changed(update)
B.on_changed(update)
C.on_changed(update)
D.on_changed(update)


resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
button = Button(resetax, 'Reset', color='gold',
                hovercolor='skyblue')


def resetSlider(event):
    A.reset()
    B.reset()
    C.reset()
    D.reset()

button.on_clicked(resetSlider)

plt.show()

#%%

#Greeks

def black_scholes_Greeks(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1) #how much the option price changes with respect to the underlying asset price
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T)) #how much the delta changes with respect to the underlying asset price
    vega = S0 * norm.pdf(d1) * np.sqrt(T) #how much the option price changes with respect to the underlying asset volatility
    theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)) #how much the option price changes with respect to the passage of time
    rho = K * T * np.exp(-r*T) * norm.cdf(d2) #how much the option price changes with respect to the interest rate

    return delta, gamma, vega, theta, rho
print(f"Delta: {black_scholes_Greeks(S0, K, r, sigma, T)[0]:.4f}")
print(f"Gamma: {black_scholes_Greeks(S0, K, r, sigma, T)[1]:.4f}")
print(f"Vega: {black_scholes_Greeks(S0, K, r, sigma, T)[2]:.4f}")
print(f"Theta: {black_scholes_Greeks(S0, K, r, sigma, T)[3]:.4f}")
print(f"Rho: {black_scholes_Greeks(S0, K, r, sigma, T)[4]:.4f}")

plt.plot(np.linspace(0.1, 1000, 1000), [black_scholes_Greeks(S0, K, r, sigma, T)[0] for S0 in np.linspace(0.1, 1000, 1000)])
plt.xlabel("Asset Price")
plt.ylabel("Delta")
plt.title("Black-Scholes Delta vs Asset Price")
plt.show()

plt.plot(np.linspace(0.1, 1000, 1000), [black_scholes_Greeks(S0, K, r, sigma, T)[1] for S0 in np.linspace(0.1, 1000, 1000)])
plt.xlabel("Asset Price")
plt.ylabel("Gamma")
plt.title("Black-Scholes Gamma vs Asset Price")
plt.show()

plt.plot(np.linspace(0.01, 1, 1000), [black_scholes_Greeks(S0, K, r, sigma, T)[2] for sigma in np.linspace(0.01, 1, 1000)])
plt.xlabel("Volatility")
plt.ylabel("Vega")
plt.title("Black-Scholes Vega vs Volatility")
plt.show()

plt.plot(np.linspace(0.01, 10, 1000), [black_scholes_Greeks(S0, K, r, sigma, T)[3] for T in np.linspace(0.01, 10, 1000)])
plt.xlabel("Time to Expiry")
plt.ylabel("Theta")
plt.title("Black-Scholes Theta vs Time to Expiry")
plt.show()

plt.plot(np.linspace(0.01, 1, 1000), [black_scholes_Greeks(S0, K, r, sigma, T)[4] for r in np.linspace(0.01, 1, 1000)])
plt.xlabel("Interest Rate")     
plt.ylabel("Rho")
plt.title("Black-Scholes Rho vs Interest Rate")
plt.show()

#comparison with numerical approximation of delta using central difference method
h = 0.01
C_plus = black_scholes_call(S0 + h, K, r, sigma, T)
C_minus = black_scholes_call(S0 - h, K, r, sigma, T)

numerical_delta = (C_plus - C_minus) / (2*h)
analytical_delta = black_scholes_Greeks(S0, K, r, sigma, T)[0]

print(f"Numerical Delta: {numerical_delta:.6f}")
print(f"Analytical Delta: {analytical_delta:.6f}")

#effect of h on numerical delta approximation
h_values = np.logspace(-1, -8, 50)

numerical_deltas = []

analytical_delta = black_scholes_Greeks(S0, K, r, sigma, T)[0]

for h in h_values:

    C_plus = black_scholes_call(S0 + h, K, r, sigma, T)
    C_minus = black_scholes_call(S0 - h, K, r, sigma, T)

    delta = (C_plus - C_minus) / (2*h)

    numerical_deltas.append(delta)

delta_errors = np.abs(np.array(numerical_deltas) - analytical_delta)

plt.plot(h_values, delta_errors)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Step size h")
plt.ylabel("Absolute Error")
plt.title("Finite Difference Error for Delta")

plt.show()

#At relatively large h, the error decreases as h gets smaller. But eventually, when h becomes very small, the error starts increasing again due to floating-point round-off error..
# %% 

#Antithetic variates method

def antithetic_monte_carlo_option_price(S0, K, r, sigma, T, num_simulations=100000):
    Z = np.random.normal(size=num_simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    ST_antithetic = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    payoff_antithetic = np.maximum(ST_antithetic - K, 0)
    price = np.exp(-r*T) * np.mean((payoff + payoff_antithetic) / 2)
    return price

#compare the standard Monte Carlo and antithetic variates method

def monte_carlo_option_price_crn(S0, K, r, sigma, T, Z):
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    return price

def antithetic_monte_carlo_option_price_crn(S0, K, r, sigma, T, Z):
    ST_pos = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    ST_neg = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * Z)
    payoff = (np.maximum(ST_pos - K, 0) + np.maximum(ST_neg - K, 0)) / 2
    price = np.exp(-r*T) * np.mean(payoff)
    return price


standard_prices = []
antithetic_prices = []

np.random.seed(42)
num_trials = 1000

for i in range(num_trials):
    Z = np.random.normal(size=1000)
    standard_prices.append(
        monte_carlo_option_price_crn(
            S0, K, r, sigma, T, Z
        )
    )

    antithetic_prices.append(
        antithetic_monte_carlo_option_price_crn(
            S0, K, r, sigma, T, Z
        )
    )

print(f"Standard MC standard deviation: {np.std(standard_prices):.4f}")
print(f"Antithetic MC standard deviation: {np.std(antithetic_prices):.4f}")

variance_reduction = (
    1 - np.var(antithetic_prices) / np.var(standard_prices)
) * 100

print(f"Variance reduction: {variance_reduction:.2f}%")

#variance is reduced by using antithetic variates method, which is a variance reduction technique. This means that the estimates of the option price are more stable and reliable when using this method compared to standard Monte Carlo simulation.
#however it should be noted that the antithetic variates method uses 1000 random draws to generate 2000 paths, as it pairs each Z with -Z. Therefore, while it reduces variance, we may wish to compare the standard Monte Carlo method with 2000 simulations to the antithetic variates method with 1000 simulations to see if the variance reduction is still significant.
#we can run this below:

standard_prices_2000 = []
np.random.seed(42)
for i in range(num_trials):
    Z_2000 = np.random.normal(size=2000)
    standard_prices_2000.append(monte_carlo_option_price_crn(S0, K, r, sigma, T, Z_2000))

print(f"Standard MC (2000 sims) standard deviation: {np.std(standard_prices_2000):.4f}")
print(f"Antithetic MC (1000 sims) standard deviation: {np.std(antithetic_prices):.4f}")
print(f"Variance still reduced by: {(1 - np.var(antithetic_prices) / np.var(standard_prices_2000)) * 100:.2f}%")

#and variance is still reduced by using the antithetic variates method, even when compared to a standard Monte Carlo simulation with double the number of simulations. This demonstrates the effectiveness of the antithetic variates method in reducing variance and improving the stability of option price estimates.

plt.hist(standard_prices_2000, bins=30, alpha=0.6, label="Standard MC (2000)")
plt.hist(antithetic_prices, bins=30, alpha=0.6, label="Antithetic MC (1000 pairs)")

plt.xlabel("Estimated Option Price (£)")
plt.ylabel("Frequency")
plt.title("Distribution of Monte Carlo Option Price Estimates")
plt.legend()
plt.show()