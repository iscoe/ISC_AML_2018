# ISC 2018 Adversarial Machine Learning Challenge 

Software for the [ISC 2018 adversarial machine learning (AML) challenge](https://challenges.jhuapl.edu/c/aml/).

For more details about the challenge (or if you have questions, comments, etc.) please visit us on [Cooler](https://cooler.jhuapl.edu/groups/profile/300279/isc-adversarial-machine-learning-challenge)!


## Quick Start

- Baseline [attacks](src/attacks) and [defenses](src/defenses) are provided.  Feel free to use these as starting points for your submissions!  Some of these may even appear as "competitors"...
- The [evaluation script](./src/evaluate_submissions.py) shows how we intend to run the attacks and defenses.  You can use this to stand up your own local simulated evaluation environment.  Note that we are using Python 3 on a Linux machine equipped with nvidia-docker.
- The [Makefile](./src/Makefile) provides examples of how to run all of these things.
