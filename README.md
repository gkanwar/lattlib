LattLib
=======
Freely usable reference code for prototyping lattice field theory calculations and analysis. All code contained in this repository is covered by the CC0 license, allowing one to freely copy, modify, or distribute anything contained herein, including for commerical purposes. Attribution not required.

Scope of the project
--------------------
This repository provides ...

* A set of Mathematica tools for simple analysis and plotting (`mathlib/`, also available in [paclet form](https://scripts.mit.edu/~gurtej/mma_paclets/qcdlib.paclet)).
* A set of Python tools for simple analysis and plotting (`pylib/`).
* Several libraries with useful functions for MCMC sampling and observable measurements in particular theories (`gauge_theory/`, `scalar_field/`, ...).
* A set of executable Python scripts that expose simulation and measurement functions for various theories (`pybin/`). These scripts generally assume the root repository directory is on the Python path.

Non-scope of the project
------------------------
Code in this repository is intended to serve as a basis for simple prototyping and explorations in low-dimensional theories, thus implementations are generally *not optimized, parallelized, or distributed*.

This code is also provided as-is, without guarantees of any kind. Use at your own risk.

Contributions
-------------
This project is not actively being developed, however contributions in the form of pull requests for bug fixes or clean-ups are welcome.

Many thanks to the following folks who have been willing test subjects, have reported bugs, and/or have contributed code!

* Michael S. Albergo ([malbergo](https://github.com/malbergo))
* Artur Avkhadiev ([avkhadiev](https://github.com/avkhadiev))
* Denis Boyda ([boydad](https://github.com/boydad))
* Anthony Grebe ([agrebe](https://github.com/agrebe))
* Daniel C. Hackett ([dchackett](https://github.com/dchackett))
* Julian M. Urban ([julian-urban](https://github.com/julian-urban))
* Michael L. Wagman ([mlwagman](https://github.com/mlwagman))
