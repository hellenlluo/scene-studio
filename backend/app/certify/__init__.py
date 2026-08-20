"""Per-axis certification and the repair operators that answer it.

Split by axis because the contract is per axis: a scene can be metrically sound
and kinematically broken, and a single validator returning one boolean would
throw away exactly the information the user needs to fix it.
"""
