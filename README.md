# fvm

Monolithic finite volume codes in 2D.

## Contents

This repository contains a handful of 2D staggered finite volume codes for simple flow problems. The implementations heavily borrow from <a href="https://github.com/saadtony/uCFD">this repository</a>.
The numerical scheme is kept as simple as possible:

- Centered or upwind fluxes
- Structured grid
- Incremental pressure correction method

## Results

| **`Lid-driven cavity`**                                 | **`Rayleigh-Benard convection`**                        |
|:-------------------------------------------------------:|:-------------------------------------------------------:|
| <img width="300" alt="" src="cavity/re_500.gif">        | <img width="300" alt="" src="rayleigh/temperature.gif"> |
| **`Passive scalar transport`**                          |                                                         |
| <img width="300" alt="" src="mixing/concentration.gif"> |                                                         |
|                                                         |                                                         |

| **`Poiseuille flow`**                                  |
|:------------------------------------------------------:|
| <img width="600" alt="" src="poiseuille/velocity.gif"> |
| **`Von Karman flow`**                                  |
| <img width="600" alt="" src="karman/velocity.gif">     |
