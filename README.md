# AeroScan — Aircraft Damage Detection & Part Risk Assessment

A two-part project:

1. **Damage Scan** — a YOLOv8 image classifier that detects whether a surface photo shows a **crack** or a **dent**.
2. **Part Risk Assessment** — a rule-based scoring tool that estimates which aircraft parts carry the highest structural risk, based on aircraft model, age, malfunction history, and past damage types. (Rule-based rather than ML, since no historical failure dataset exists yet — see *Notes* below.)

Both are wrapped in a single-page web dashboard (`aircraft_dashboard.html`).

---

## Project structure

```
AERO_PROJECT/
│
├── aircraft_damage_dataset_v1/       # your image dataset
│   ├── train/{crack,dent}/
│   ├── val/{crack,dent}/             # renamed from "valid"
│   └── test/{crack,dent}/
│
├── aircraft_damage_project.ipynb     # trains the classifier
│
├── runs_damage/                      # created after training
│   └── crack_dent_cls/weights/best.pt
│
├── requirements.txt
├── README.md
│
└── webapp/
    ├── aircraft_dashboard.html       # front end — open this in a browser
    └── app.py                        # backend — serves predictions from best.pt
```

---

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Rename the validation folder** (Ultralytics expects `val`, not `valid`):
   ```bash
   mv aircraft_damage_dataset_v1/valid aircraft_damage_dataset_v1/val
   ```

3. **Train the classifier**
   Open `aircraft_damage_project.ipynb` in Jupyter and run all cells. This creates:
   ```
   runs_damage/crack_dent_cls/weights/best.pt
   ```

4. **Point the backend at your trained model**
   Make sure `runs_damage/` is reachable from wherever you run `app.py`. Either:
   - copy `runs_damage/` into `webapp/`, **or**
   - edit `MODEL_PATH` in `app.py` to the correct relative/absolute path.

5. **Run the backend**
   ```bash
   cd webapp
   python app.py
   ```
   Leave this running — it serves predictions at `http://localhost:5000/predict`.

6. **Open the dashboard**
   Open `webapp/aircraft_dashboard.html` directly in a browser (double-click it, no server needed for the page itself).

---

## Using the dashboard

- **Damage Scan panel** — upload a photo, click "Scan for damage." Requires `app.py` to be running (step 5). If it's not running, the page will tell you plainly instead of failing silently.
- **Part Risk Assessment panel** — enter aircraft model, year built, malfunction count, and any past damage types, then click "Assess risk." This runs entirely in the browser — no backend needed.

---

## Notes on the risk assessment logic

There's no historical maintenance/failure dataset behind the risk scores — a real ML model would need that to train on. Instead, `predictRisk()` (in the `<script>` block of `aircraft_dashboard.html`) uses a transparent, hand-set scoring system based on general aviation engineering heuristics: aircraft age, an estimated usage/cycle proxy, and known links between past damage types and nearby parts.

This is meant to be a credible **starting point for a pitch**, not a production risk model. The honest next step: once real maintenance/inspection data is available, replace `predictRisk()` with a model trained on that data.

---

## Retraining with more data

If you get more labeled images later, just re-run `aircraft_damage_project.ipynb` — it will retrain and overwrite `best.pt` with the improved model. No other files need to change.
