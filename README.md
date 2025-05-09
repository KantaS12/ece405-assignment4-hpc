# UHM ECE 496B Spring 2025 Assignment 4: HPC

This asignment is created from Assignment 2 of [CS336 at Stanford taught in Spring 2024](https://stanford-cs336.github.io/spring2024/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2024_assignment2_systems.pdf](./cs336_spring2024_assignment2_systems.pdf)

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2024-lectures), especially [Lecture 8](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_08.py).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix. Any improvements of the existing codebase
(including adaptations from Stanford to UHM workflows, modifications of PDF, etc) will be rewarded with extra points.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `setup.py`. This module should contain
  your from-scratch language model from assignment 1.
- [`./cs336-systems`](./cs336-systems): directory containing a module
  `cs336_systems` and its associated `setup.py`. In this module, you will
  implement an optimized Transformer language model---feel free to take your
  code from assignment 1 (in `cs336-basics`) and copy it over as a starting
  point. In addition, you will implement for distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336-basics # Files from assignment 1 
│   ├── cs336_basics # A python module named cs336_basics
│   │   ├── __init__.py
│   │   ├── VERSION
│   │   └── ... other files in the cs336_basics module, taken from assignment 1 ...
│   ├── requirements.txt
│   └── setup.py (setup.py to install `cs336_basics`) 
├── cs336-systems # TODO(you):code that you'll write for assignment 2 
│   ├── cs336_systems # A python module named cs336_systems
│   │   ├── __init__.py
│   │   ├── VERSION
│   │   └── ... TODO(you): other python files that you need for assignment 2 ...
│   ├── requirements.txt
│   ├── ... TODO(you): any other files or folders you need for assignment 2 ...
│   └── setup.py (setup.py to install `cs336_systems`)
├── README.md
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

0. Set up a conda environment and install packages. In particular, the
   `cs336-basics` package (located at [`./cs336-basics`](./cs336-basics))
   installs the `cs336_basics` module, and the `cs336-systems` package (located
   at [`./cs336-systems`](./cs336-systems)) installs the `cs336_systems` module.

``` sh
conda create -n cs336_systems python=3.10 --yes
conda activate cs336_systems
pip install -e ./cs336-basics/ -e ./cs336-systems/'[test]'
```

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.


## ECE491B Assignment instructions

Follow along the [CS336@Stanford handout](./cs336_spring2024_assignment2_systems.pdf) with small deviations:
1. What the code looks like: clone https://github.com/igormolybog/s2025-assignment4-hpc
2. How to submit: You will submit the report on the assignment to [Assignment Submission Form](https://forms.gle/CSRweWjuBxvYbb9MA). The code does not have to be attached as long as you include links to the main GitHub branch where your code lives and links to all of the Colab notebooks if applicable.
3. Section 2 and related work can be performed in Colab pr locally. Section 3 has to be performed using [Koa cluster](https://docs.google.com/document/d/1h00x2pAjIjMDJ-1RBeHQaTvnfxUhM_lAVNbskEc9f7A/edit?usp=sharing).
4. In case the large models (XL and 2.7B) result in out of memory errors (OOM), feel free to do profiling experiments with smaller models.
5. Koa cluster does not support multi-node communication. Skip Sections 3.2, 3.4.3, and 4 and associated Problems. In the rest of the problems, multi-node communication should be replaced with a single-node multi-GPU setup.
6. What you can use: Implementation from scratch is preferred, but experiments are essential. If you are stuck with some implementation, just use the Huggingface/Pytorch implementation and proceed to the experiments.
    - Submit the report reflecting your attempts at implementation for partial credit
