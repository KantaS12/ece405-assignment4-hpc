# UHM ECE 405 Spring 2026 Assignment 4: HPC

This asignment is created from Assignment 2 of [CS336 at Stanford taught in Spring 2025](https://stanford-cs336.github.io/spring2025/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2025_assignment2_systems.pdf](./cs336_spring2025_assignment2_systems.pdf)

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2025-lectures), especially [Lecture 8](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_08.py).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix. Any improvements of the existing codebase
(including adaptations from Stanford to UHM workflows, modifications of PDF, etc) will be rewarded with extra points.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.12.10
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 85 packages in 711ms
Python 3.12.10 (main, Apr  9 2025, 04:03:51) [Clang 20.1.0 ] on linux
...
>>> import cs336_basics
>>> 
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.


## ECE405 Assignment instructions

Follow along the [CS336@Stanford handout](./cs336_spring2025_assignment2_systems.pdf) with small deviations:
1. What the code looks like: clone https://github.com/igormolybog/s2025-assignment4-hpc
2. How to submit: You will submit the report on the assignment to [Assignment Submission Form](https://docs.google.com/forms/d/e/1FAIpQLScJg_QkwjKux3xKeM-EOmZyvA6zlbVIrf_lxN_qoCFoxdqTrg/viewform?usp=sharing&ouid=111841773839267096112). The code does not have to be attached as long as you include links to the main GitHub branch where your code lives and links to all of the Colab notebooks if applicable.
3. Section 2 and related work can be performed in Colab pr locally. Section 3 has to be performed using [Koa cluster](https://docs.google.com/document/d/1JrDvkOQw6GKHKgzk6qX1UsVjvLHuqEb9sfnDtU4JL2I/edit?usp=sharing).
4. In case the large models (XL and 2.7B) result in out of memory errors (OOM), feel free to do profiling experiments with smaller models.
5. Koa cluster does not support multi-node communication. Skip Sections 3.2, 3.4.3, and 4 and associated Problems. In the rest of the problems, multi-node communication should be replaced with a single-node multi-GPU setup.
6. What you can use: Implementation from scratch is preferred, but experiments are essential. If you are stuck with some implementation, just use the Huggingface/Pytorch implementation and proceed to the experiments.
    - Submit the report reflecting your attempts at implementation for partial credit
