#!/usr/bin/env python3
"""
IEM EarPrint Engine
Deterministic implementation of the supplied EarPrint prompt with explicit math-lock definitions.

SOURCE-OF-TRUTH MATH
--------------------
This engine follows the supplied prompt for:
- one independent IEM = one equal vote
- log-frequency interpolation onto one declared master grid
- three alignment scenarios: [200,1000], [200,500], [500,1000] Hz
- per-IEM scenario centre = pointwise median of the 3 aligned scenarios
- per-IEM alignment uncertainty(f) = pointwise max absolute distance from that centre
- pure EarPrint construction and single Gaussian smoothing pass
- target-specific robust masking with frequency-dependent AlignmentUncertainty
- conditional RepeatabilityFloor: inactive when no valid same-IEM repeats exist
- retention / raw mask / sin^2 boundary taper / domain-zero rules
- dynamic target discovery and one hybrid per discovered target as an explicit extension
  of the prompt's two named hybrids.

Boundary rule:
- lower/start personal-domain transition: configurable sin^2, locked to 1/3 octave
- upper/end personal-domain transition: configurable sin^2, retained at 1/4 octave

No target list or IEM list is hard-coded.
The engine accepts authoritative preferred-response curves as its sole calculation input.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d