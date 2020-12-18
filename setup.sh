#!/bin/bash
#
# ~~~
# This file is part of the paper:
#
#           "An adaptive projected Newton non-conforming dual approach
#         for trust-region reduced basis approximation of PDE-constrained
#                           parameter optimization"
#
#   https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt
#
# Copyright 2019-2020 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Felix Schindler (2020)
#   Tim Keil        (2020)
# ~~~

set -e

# initialize the virtualenv
export BASEDIR="${PWD}"
virtualenv --python=python3 venv
source venv/bin/activate

# install python dependencies into the virtualenv
cd "${BASEDIR}"
pip install --upgrade pip
pip install $(grep Cython requirements.txt)
pip install -r requirements.txt

# install local pymor and pdeopt version
cd "${BASEDIR}"
cd pymor && pip install -e .
cd "${BASEDIR}"
cd pdeopt && pip install -e .

cd "${BASEDIR}"
echo
echo "All done! From now on run"
echo "  source venv/bin/activate"
echo "to activate the virtualenv!"
