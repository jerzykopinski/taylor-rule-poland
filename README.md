This script estimates a static Taylor rule for Poland using quarterly macroeconomic data.

It compares the **actual NBP policy rate** with a **model-implied benchmark rate** based on inflation and economic activity.

---

MODEL SPECIFICATION:
i_t = c + alpha * (pi_t - pi_target_t) + beta * output_gap_t + epsilon_t

where:
- i_t is the NBP policy rate
- pi_t is quarterly inflation
- pi_target_t is the inflation target
- output_gap_t is the quarterly output gap

---

What this does

- Builds a quarterly dataset for Poland (data is included):
  - inflation (HICP, YoY)
  - output gap (HP filter on real GDP)
  - policy rate (NBP reference rate)
- Estimates a Taylor rule using OLS
- Computes the implied “equilibrium” interest rate
- Compares it to the actual policy rate
- Prepares a easy to understand visualizations

---

How to run

From the project root:

```bash
python src/main.py
```

Enjoy!
