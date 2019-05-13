# confidence_intervals
Confidence intervals for classification performance metrics. Requires numpy and scipy.stats.

Function normal_approximation_binomial_confidence_interval should work for any metric computed as 
a ratio of events/observations, such as TPR, TNR, FPR, FNR, precision, or negative predictive value.
It uses the normal approximation of a binomial confidence interval.

Function f1_score_confidence_interval does the same, but for the F1-score, with the propagated uncertainties
of recall and precision, using an expression that I derived (confirmation appreciated!).

Here's an example for: true positives = 42, real positives = 63, predicted positives = 69.


ipython

Python 3.7.2 (default, Dec 29 2018, 06:19:36) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import propagation_confidence_interval as prop

In [2]: prop.normal_approximation_binomial_confidence_interval?
Signature: prop.normal_approximation_binomial_confidence_interval(s, n, confidence_level=0.95)
Docstring:
Computes the binomial confidence interval of the probability of a success s, 
based on the sample of n observations. The normal approximation is used,
appropriate when n is equal to or greater than 30 observations.
The confidence level is between 0 and 1, with default 0.95.
Returns [p_estimate, interval_range, lower_bound, upper_bound].
For reference, see Section 5.2 of Tom Mitchel's "Machine Learning" book.
File:      ~/stluc-experiments/scripts/propagation_confidence_interval.py
Type:      function

In [3]: recall_successes = 42

In [4]: recall_obs = 63

In [5]: [r, dr, r_upper_bound, r_lower_bound] = prop.normal_approximation_binomial_confidence_interval(recall_successes, recall_obs)

In [6]: r
Out[6]: 0.6666666666666666

In [7]: dr
Out[7]: 0.1164049796915108

In [8]: precision_successes = 42

In [9]: precision_obs = 69

In [10]: [p, dp, p_upper_bound, p_lower_bound] = prop.normal_approximation_binomial_confidence_interval(precision_successes, precision_obs)

In [11]: p
Out[11]: 0.6086956521739131

In [12]: dp
Out[12]: 0.1151545180912738

In [13]: prop.f1_score_confidence_interval?
Signature: prop.f1_score_confidence_interval(r, p, dr, dp)
Docstring:
Computes the confidence interval for the F1-score measure of classification performance
based on the values of recall (r), precision (p), and their respective confidence
interval ranges, or absolute uncertainty, about the recall (dr) and the precision (dp).
Disclaimer: I derived the formula myself based on f(r,p) = 2rp / (r+p).
Nobody has revised my computation. Feedback appreciated!
File:      ~/stluc-experiments/scripts/propagation_confidence_interval.py
Type:      function

In [14]: prop.f1_score_confidence_interval(r, p, dr, dp)
Out[14]: 
(0.6363636363636365,
 0.18307035592733717,
 0.4532932804362993,
 0.8194339922909737)
