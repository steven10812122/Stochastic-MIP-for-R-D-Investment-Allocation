import numpy as np
import pulp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# ============================================================
#  R&D Investment Allocation
#  Stochastic Mixed-Integer Program with Chance Constraints
#
#  Formal model:
#
#    max   Σ_j  α_j · z_j
#
#    s.t.  Σ_j x_j              ≤ B                    (budget)
#          x_j                  ≤ cap_j · y_j  ∀j       (capacity + activation)
#          x_j                  ≥ c_j   · y_j  ∀j       (minimum investment)
#          z_j  ≤ cap_j · y_j                           (McCormick 1)
#          z_j  ≤ x_j                                   (McCormick 2)
#          z_j  ≥ x_j - cap_j·(1-y_j)                  (McCormick 3)
#          z_j  ≥ 0                                     (McCormick 4)
#
#          Σ_j α_j·z_j  −  Φ⁻¹(1-ε) · √(Σ_j σ_j²·y_j)  ≥  R_min
#                                                        (Chance Constraint)
#          y_j ∈ {0,1},   x_j ≥ 0
#
#  Chance Constraint (deterministic equivalent):
#    P( Σ r̃_j  ≥  R_min )  ≥  1 − ε
#    ⟺  E[return] − Φ⁻¹(1−ε) · σ[return]  ≥  R_min
#
#  Key insight from σ design:
#    AI Research and Robotics have high expected return BUT high σ.
#    When ε is small (strict CC), the Φ⁻¹(1−ε) penalty is large,
#    pushing the solver toward low-σ portfolios (Manufacturing, Green Tech).
#    When ε is large (relaxed CC), high-return / high-σ projects win.
#    This creates a meaningful policy switch across ε values.
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# 1. Problem Data
# ------------------------------------------------------------

projects = ["AI Research", "Manufacturing", "Healthcare", "Green Tech", "Robotics"]
J        = len(projects)

#           AI     MFG    HC     Green  Rob
alpha  = np.array([0.30,  0.20,  0.22,  0.18,  0.25])  # return per $ invested
sigma  = np.array([1.20,  0.05,  0.15,  0.10,  0.90])  # std of stochastic return
c_min  = np.array([10.0,  8.0,   6.0,   5.0,   9.0])   # minimum activation cost
cap    = np.array([15.0,  20.0,  15.0,  12.0,  18.0])  # max investment per project

B     = 40.0   # Total budget
R_min = 7.5    # Minimum acceptable return (chance constraint RHS)
#
# With R_min = 7.5:
#   ε = 0.01 (strict)  → Φ⁻¹(0.99)=2.33, heavy σ penalty
#                         only low-σ portfolio (MFG+HC+Green) is feasible
#   ε = 0.30 (relaxed) → Φ⁻¹(0.70)=0.52, light σ penalty
#                         high-return portfolio (AI+Robotics+HC) becomes optimal


# ------------------------------------------------------------
# 2. Core Solver
#
#    Enumerate all non-empty project subsets (2^J − 1 = 31).
#    For each feasible subset y:
#      - Compute chance constraint RHS
#      - Solve LP over continuous x (y fixed → McCormick collapses)
#      - Track best feasible objective
# ------------------------------------------------------------

def solve(epsilon):
    z_cc      = norm.ppf(1 - epsilon)   # Φ⁻¹(1−ε)
    best_obj  = -np.inf
    best_x    = None
    best_y    = None

    for mask in range(1, 2**J):
        y         = np.array([(mask >> j) & 1 for j in range(J)], dtype=float)
        total_std = np.sqrt(np.dot(sigma**2, y))
        cc_rhs    = R_min + z_cc * total_std   # required E[return] ≥ this

        # Skip if even max possible return can't satisfy CC
        if np.dot(alpha * cap, y) < cc_rhs:
            continue

        prob = pulp.LpProblem("LP", pulp.LpMaximize)
        x    = [pulp.LpVariable(f"x{j}", lowBound=0) for j in range(J)]

        prob += pulp.lpSum(alpha[j] * x[j] for j in range(J))
        prob += pulp.lpSum(x[j] for j in range(J)) <= B

        for j in range(J):
            if y[j] == 1:
                prob += x[j] >= c_min[j]
                prob += x[j] <= cap[j]
            else:
                prob += x[j] == 0

        # Deterministic equivalent of chance constraint
        prob += pulp.lpSum(alpha[j] * x[j] for j in range(J)) >= cc_rhs

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == "Optimal":
            obj = pulp.value(prob.objective)
            if obj > best_obj:
                best_obj = obj
                best_x   = np.array([pulp.value(x[j]) for j in range(J)])
                best_y   = y.copy()

    return (best_obj, best_x, best_y) if best_x is not None else (None, None, None)


# ------------------------------------------------------------
# 3. Base Case  (ε = 0.10)
# ------------------------------------------------------------

EPSILON = 0.10

print("=" * 62)
print("  Stochastic MIP — R&D Investment Allocation")
print(f"  Chance Constraint: P(return ≥ {R_min}) ≥ {1-EPSILON:.0%}  (ε={EPSILON})")
print("=" * 62)

obj, x_opt, y_opt = solve(EPSILON)
z_base     = norm.ppf(1 - EPSILON)
total_std  = np.sqrt(np.dot(sigma**2, y_opt))

print(f"\n  Status    : Optimal")
print(f"  E[Return] : {obj:.4f}\n")
print(f"  {'Project':<18} {'y_j':>5} {'Cap':>6} {'x_j':>10} {'α·x_j':>8} {'σ_j':>6}")
print("  " + "-" * 58)
for j in range(J):
    print(f"  {projects[j]:<18} {int(y_opt[j]):>5} {cap[j]:>6.0f} "
          f"{x_opt[j]:>10.2f} {alpha[j]*x_opt[j]:>8.4f} {sigma[j]:>6.2f}")
print(f"\n  Budget used    : {x_opt.sum():.2f} / {B:.2f}")
print(f"  Portfolio σ    : {total_std:.4f}")
print(f"  Φ⁻¹(1−ε)      : {z_base:.4f}")
print(f"  CC LHS         : {obj - z_base*total_std:.4f}  ≥  {R_min}  ✓")


# ------------------------------------------------------------
# 4. Sensitivity Analysis — policy switches across ε
# ------------------------------------------------------------

epsilons = [0.01, 0.05, 0.10, 0.20, 0.30]
sens     = []

print("\n" + "=" * 62)
print("  Sensitivity Analysis — varying ε")
print("=" * 62)
print(f"\n  {'ε':>5}  {'1−ε':>6}  {'E[R]':>7}  {'σ':>7}  Projects selected")
print("  " + "-" * 62)

for eps in epsilons:
    o, xv, yv = solve(eps)
    if o is not None:
        s   = np.sqrt(np.dot(sigma**2, yv))
        sel = [projects[j] for j in range(J) if yv[j]]
        sens.append((eps, o, s, xv, yv))
        print(f"  {eps:>5.2f}  {1-eps:>5.0%}  {o:>7.4f}  {s:>7.4f}  {', '.join(sel)}")
    else:
        sens.append((eps, None, None, None, None))
        print(f"  {eps:>5.2f}  {1-eps:>5.0%}  {'Infeasible':>7}")


# ------------------------------------------------------------
# 5. Monte Carlo Validation  (base case ε = 0.10)
# ------------------------------------------------------------

N_MC  = 50_000
xi    = np.random.normal(0, sigma, size=(N_MC, J))
r_mc  = np.sum((alpha * x_opt + xi) * y_opt, axis=1)
emp_p = np.mean(r_mc >= R_min)

print("\n" + "=" * 62)
print(f"  Monte Carlo Validation  (N = {N_MC:,},  ε = {EPSILON})")
print("=" * 62)
print(f"\n  Required   P(return ≥ {R_min}) ≥ {1-EPSILON:.0%}")
print(f"  Empirical  P(return ≥ {R_min})  = {emp_p:.2%}  ✓")
print(f"  Mean  : {r_mc.mean():.4f}")
print(f"  σ     : {r_mc.std():.4f}")
print(f"  10th% : {np.percentile(r_mc,10):.4f}")


# ------------------------------------------------------------
# 6. Visualisation
# ------------------------------------------------------------

C = ["#3B82C4", "#E05C2E", "#27A96C", "#9B59B6", "#E67E22"]

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Stochastic MIP — R&D Investment Allocation",
             fontsize=14, fontweight="bold", y=0.99)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)


# ── Panel A: Base-case allocation ─────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
sel  = [j for j in range(J) if y_opt[j]]
bars = ax_a.bar([projects[j] for j in sel],
                [x_opt[j] for j in sel],
                color=[C[j] for j in sel],
                edgecolor="white", linewidth=1.2, width=0.5)
for bar, j in zip(bars, sel):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
              f"E[r]={alpha[j]*x_opt[j]:.2f}",
              ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
              f"σ={sigma[j]:.2f}",
              ha="center", va="center", fontsize=9, color="white", fontweight="bold")
ax_a.set_title(f"A. Optimal Allocation  (ε={EPSILON}, service={1-EPSILON:.0%})",
               fontweight="bold")
ax_a.set_ylabel("Investment ($)")
ax_a.set_ylim(0, max(cap)*1.25)
ax_a.grid(axis="y", alpha=0.3)


# ── Panel B: Policy map across ε ──────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

valid = [(e, o, s, xv, yv) for e, o, s, xv, yv in sens if o is not None]
eps_v = [v[0] for v in valid]
obj_v = [v[1] for v in valid]
std_v = [v[2] for v in valid]

ax_b.plot(eps_v, obj_v, "o-", color="#3B82C4", lw=2, ms=7, label="E[Return]")
ax_b2 = ax_b.twinx()
ax_b2.plot(eps_v, std_v, "s--", color="#E05C2E", lw=1.5, ms=6, label="Portfolio σ")

# Annotate policy switch
prev_sel = None
for eps, o, s, xv, yv in valid:
    cur_sel = tuple(j for j in range(J) if yv[j])
    if prev_sel is not None and cur_sel != prev_sel:
        ax_b.axvline(eps, color="gray", lw=1, linestyle=":")
        ax_b.text(eps+0.002, min(obj_v)+0.1, "Policy\nswitch",
                  fontsize=7.5, color="gray")
    prev_sel = cur_sel

ax_b.set_xlabel("ε  (allowed violation probability)")
ax_b.set_ylabel("E[Return]", color="#3B82C4")
ax_b2.set_ylabel("Portfolio σ", color="#E05C2E")
ax_b.tick_params(axis="y", colors="#3B82C4")
ax_b2.tick_params(axis="y", colors="#E05C2E")
ax_b.set_title("B. Sensitivity: ε vs Return & Risk\n(vertical line = policy switch)",
               fontweight="bold")
l1,lb1 = ax_b.get_legend_handles_labels()
l2,lb2 = ax_b2.get_legend_handles_labels()
ax_b.legend(l1+l2, lb1+lb2, fontsize=8.5, loc="upper left")
ax_b.grid(alpha=0.3)


# ── Panel C: Monte Carlo validation ───────────────────────
ax_c = fig.add_subplot(gs[1, :])

ax_c.hist(r_mc, bins=120, density=True, color="#3B82C4", alpha=0.65,
          label=f"Sampled return  (N={N_MC:,})")

mu_th = float(np.dot(alpha * x_opt, y_opt))
x_th  = np.linspace(r_mc.min(), r_mc.max(), 500)
ax_c.plot(x_th, norm.pdf(x_th, mu_th, total_std), color="#1a1a2e", lw=2,
          label=f"Theoretical  N({mu_th:.2f}, {total_std:.2f}²)")

ax_c.axvline(R_min, color="#E05C2E", lw=2, label=f"R_min = {R_min}")
p10 = np.percentile(r_mc, 10)
ax_c.axvline(p10, color="#E05C2E", lw=1.5, ls="--",
             label=f"Empirical 10th pct = {p10:.2f}")

x_sh = np.linspace(r_mc.min(), R_min, 300)
ax_c.fill_between(x_sh, norm.pdf(x_sh, mu_th, total_std),
                  alpha=0.25, color="#E05C2E",
                  label=f"Violation  P = {1-emp_p:.2%}")

ax_c.set_title(
    f"C. Monte Carlo Validation  (N={N_MC:,})\n"
    f"Empirical P(return ≥ {R_min}) = {emp_p:.2%}  "
    f"[required ≥ {1-EPSILON:.0%}]",
    fontweight="bold")
ax_c.set_xlabel("Total stochastic return")
ax_c.set_ylabel("Density")
ax_c.legend(fontsize=9)
ax_c.grid(alpha=0.3)

plt.savefig("rd_stochastic_mip_steven.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  Figure saved → rd_stochastic_mip.png")
print("Done.")
