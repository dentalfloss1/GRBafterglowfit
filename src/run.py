import yaml  # 📄 config loader
import numpy as np  # 🔢 math engine
import matplotlib.pyplot as plt  # 📊 plotting
import corner  # 🔺 posterior visualization

from data import load_data, prepare_fit_data  # 📥 data pipeline
from fit import run_mcmc, make_model  # 🔧 fitting machinery
import os  # 📂 file system checks
import sys  # 🚪 clean exit

def create_template_config(path="config.yaml"):
    print("📝 No config.yaml found — creating template...")

    template = """# 🔧 GRB Fit Configuration File

burst:
  name: your_grb_name_here
  t0: 1.0

data:
  radio_file: grbmeas.csv
  other_file: otherdata.txt
  batxrt_file: batxrt.txt

model:
  type: forward_reverse   # options: forward_only, forward_reverse
  k: 2
  p: 2.2

fit:
  initial_guess:
    f0: 1e-3
    f0_rev: 5e-5
    nua0_rev: 10
    nua_0: 13
    num_0: 100
    nuc_0: 8e8

  bounds:
    f0: [1e-6, 1]
    f0_rev: [3e-5, 1e5]
    nua0_rev: [0.1, 1e6]
    nua_0: [6, 15]
    num_0: [33, 1e4]
    nuc_0: [1e8, 2e9]
"""

    with open(path, "w") as f:
        f.write(template)

    print(f"✅ Template written to {path}")
    print(f"👉 Please fill out the fit parameters in '{path}' and rerun.\n")

# 🧠 Summarize posterior distributions
def summarize_chain(samples, keys):
    print("\n=== 📊 POSTERIOR SUMMARY ===")
    for i, k in enumerate(keys):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])  # 📏 credible intervals
        q = np.diff(mcmc)
        print(f"{k} = {mcmc[1]:.4e} +{q[1]:.4e} -{q[0]:.4e}")


# 🔺 Corner plot (parameter correlations + constraints)
def plot_corner(samples, keys):
    print("📈 Generating corner plot...")
    fig = corner.corner(samples, labels=keys, show_titles=True)
    plt.savefig("corner.png", dpi=200)  # 💾 save plot
    plt.close()
    print("✅ corner.png saved")


# 🎯 Posterior predictive visualization (the “spaghetti plot” 🍝)
def plot_posterior_models(cfg, samples, xdata, ydata, yerr, n_draws=100):
    print("🎨 Generating posterior predictive plot...")

    model = make_model(cfg)

    t, nu = xdata  # ⏱️ time + 📡 frequency

    # 🔄 smooth curve grid
    t_line = np.geomspace(t.min(), t.max(), 200)

    # 🎯 representative frequency (can expand later)
    nu_med = np.median(nu)
    nu_line = np.full_like(t_line, nu_med)

    plt.figure(figsize=(8, 6))

    # 🎲 draw random posterior samples
    inds = np.random.randint(len(samples), size=n_draws)

    for i in inds:
        theta = samples[i]  # 🎯 one parameter set
        y_model = model(theta, (t_line, nu_line)) * 1e6  # 🔄 model eval
        plt.plot(
            t_line,
            y_model,
            color="navy",
            alpha=0.05  # 👻 transparency → shows density
        )

    # 📡 plot observed data
    plt.errorbar(
        t,
        ydata * 1e6,
        yerr=yerr * 1e6,
        fmt="o",
        color="black",
        label="data 📡"
    )

    # 📏 axes scaling
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Days ⏱️")
    plt.ylabel("Flux ($\\mu$Jy) 💡")
    plt.title("Posterior predictive samples 🍝")

    plt.legend()
    plt.tight_layout()

    plt.savefig("posterior_models.png", dpi=200)  # 💾 save
    plt.close()

    print("✅ posterior_models.png saved")


# 🚀 Main execution pipeline
def main():
    print("🚀 Starting GRB fit pipeline...")

    # 📂 check for config file
    config_path = "config.yaml"

    if not os.path.exists(config_path):
        create_template_config(config_path)
        sys.exit(0)  # 🚪 exit cleanly

    # 📄 load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("📄 Config loaded")
    # 📥 load + prep data
    df = load_data(cfg)
    print(f"📊 Loaded {len(df)} data points")

    xdata, ydata, yerr = prepare_fit_data(df)
    print("🧹 Data prepared for fitting")

    # 🔥 run MCMC
    print("🔥 Running MCMC sampler...")
    keys, sampler = run_mcmc(cfg, xdata, ydata, yerr)
    print("✅ MCMC complete")

    # ✂️ burn-in removal
    print("✂️ Removing burn-in + thinning chain...")
    flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
    print(f"📦 Final sample size: {len(flat_samples)}")

    # 🧠 summarize posterior
    summarize_chain(flat_samples, keys)

    # 🔺 corner plot
    plot_corner(flat_samples, keys)

    # 🍝 posterior predictive plot
    plot_posterior_models(cfg, flat_samples, xdata, ydata, yerr)

    print("🎉 All done!")


# ▶️ entry point
if __name__ == "__main__":
    main()
