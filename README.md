# CDLEEDS
This repository contains an implementation of the CDLEEDS local change detection framework introduced in

*Johannes Haug, Alexander Braun, Stefan Zürn, and Gjergji Kasneci. 
"Change Detection for Local Explainability in Evolving Data Streams." 
31st International Conference on Information and Knowledge Management (CIKM'22). 2022. [arXiv](https://doi.org/10.48550/arXiv.2209.02764).*

CDLEEDS can serve as a model-agnostic **extension of local attribution methods in data stream applications**,
and/or as an effective **local and global concept drift detection model**.

## Use CDLEEDS for Local and Global Change Detection

CDLEEDS can be integrated into existing online learning processes as demonstrated in the example below.

We provide 3 different baseline options below. In general, however, CDLEEDS may also be used with other (custom) baselines.

The implementation provided here internally uses the RBF-Kernel as distance measure. Hence, it can only be used with
numeric data.

```python
from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from cdleeds import CDLEEDS
import numpy as np

# Load data as scikit-multiflow FileStream.
stream = FileStream('yourData.csv', target_idx=0)

# Initialize CDLEEDS and the online learning model.
detector = CDLEEDS()
model = HoeffdingTreeClassifier()
x_init, y_init = stream.next_sample(batch_size=100)
model.partial_fit(x_init, y_init, classes=stream.target_values)

# Specify a baseline (3 exemplary options):
# -------
# B1 - Zero (Constant) Baseline
# baseline = np.zeros((1, x_init.shape[1]))
# detector.set_baseline(model.predict_proba(baseline))
# -------
# B2 - Decay Window / EWMA (as described in the paper)
# B3 - Fixed-Size Sliding Window
baseline = np.mean(x_init, axis=0).reshape(1, -1)
detector.set_baseline(model.predict_proba(baseline))

# Specify a sample of observations that shall be automatically monitored for local change.
# Observations can be added and deleted dynamically using CDLEEDS().add_to_monitored_sample() 
# and CDLEEDS().delete_from_monitored_sample().
detector.add_to_monitored_sample(x_init)

while stream.has_more_samples():
    x, y = stream.next_sample(batch_size=1)  # Note: CDLEEDS also works with larger batch sizes.
    
    # Update CDLEEDS.
    detector.partial_fit(x, model.predict_proba(x))
    
    # Check for local change.
    local_drifts, centroids, monitored_obs_subject_to_change = detector.detect_local_change()
    print("There are {} local neighborhoods (i.e., leaf nodes) in total.".format(centroids.shape[0]))
    print("There are {} neighborhoods that currently underlie local change.".format(np.count_nonzero(local_drifts)))
    
    # Optional: Recompute outdated attributions.
    # for obs in monitored_obs_subject_to_change:
        # new_attr = attribution_model(obs, model)
    
    # Check for global change.
    global_drift = detector.detect_global_change()
    if global_drift:
        print("There is an ongoing global change.")
        # Optional: Reset the online learning model.
        # model = HoeffdingTreeClassifier()
        
    # Update the baseline.
    # -------
    # B1 - Zero
    # detector.set_baseline(model.predict_proba(baseline))
    # -------
    # B2 - EWMA
    baseline = 0.001 * x + (1-0.001) * baseline
    detector.set_baseline(model.predict_proba(baseline))
    # -------
    # B3 - Sliding Window (we use the centroid of the root node, ...
    # which is based on a fixed-size sliding window of recent observations.)
    # baseline = detector.root.centroid.reshape(1, -1)
    # detector.set_baseline(model.predict_proba(baseline))
    
    # Update the predictive model
    model.partial_fit(x, y, classes=stream.target_values)

stream.restart()
```
