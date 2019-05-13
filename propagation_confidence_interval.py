import numpy as np
from scipy.stats import norm

# Requires numpy and scipy.stats

def normal_approximation_binomial_confidence_interval(s, n, confidence_level=.95):
	'''Computes the binomial confidence interval of the probability of a success s, 
	based on the sample of n observations. The normal approximation is used,
	appropriate when n is equal to or greater than 30 observations.
	The confidence level is between 0 and 1, with default 0.95.
	Returns [p_estimate, interval_range, lower_bound, upper_bound].
	For reference, see Section 5.2 of Tom Mitchel's "Machine Learning" book.'''

	p_estimate = (1.0 * s) / n

	interval_range = norm.interval(confidence_level)[1] * np.sqrt( (p_estimate * (1-p_estimate))/n )

	return p_estimate, interval_range, p_estimate - interval_range, p_estimate + interval_range


def f1_score_confidence_interval(r, p, dr, dp):
	'''Computes the confidence interval for the F1-score measure of classification performance
	based on the values of recall (r), precision (p), and their respective confidence
	interval ranges, or absolute uncertainty, about the recall (dr) and the precision (dp).
	Disclaimer: I derived the formula myself based on f(r,p) = 2rp / (r+p).
	Nobody has revised my computation. Feedback appreciated!'''

	f1_score = (2.0 * r * p) / (r + p)

	left_side = np.abs( (2.0 * r * p) / (r + p) )

	right_side = np.sqrt( np.power(dr/r, 2.0) + np.power(dp/p, 2.0) + ((np.power(dr, 2.0)+np.power(dp, 2.0)) / np.power(r + p, 2.0)) )

	interval_range = left_side * right_side

	return f1_score, interval_range, f1_score - interval_range, f1_score + interval_range
