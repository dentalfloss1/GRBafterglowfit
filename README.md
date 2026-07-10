If you use this code, please cite or drop a link to this GitHub page. 

Based on modeling code originally written for Chastain, S. I., et al. (2026). “GRB 240205B: A Reverse Shock Detected in Rapid Response Radio Observations.” arXiv:2603.19047 [astro-ph.HE]. Submitted to Publications of the Astronomical Society of Australia.

This code uses the standard GRB afterglow models in either an ISM-like ($k=0$) or stellar wind-like ($k=2$) medium to give the expected temporal and spectral indices. These are then fit to the data assuming a value of $p$. The outputs are the observational parameters, not physical parameters: $F_0$, $\nu_a$, $\nu_m$, $\nu_c$, etc. This is done in order to provide fits to data even when there is not enough data to find the physical parameters.
