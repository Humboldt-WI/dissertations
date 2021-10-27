# Mitigating Covariate Shift for Low Dimensional Machine Learning Problems via Lattice Based Models 
 
## About this project
Bachelor Thesis submitted to the School of Business and Economics of Humboldt-Universität zu Berlin for the degree B.Sc. Economics.

### Built with

* [R 4.0.3](https://www.r-project.org/)
* [Python 3.8](https://www.python.org/)
* [Tensorflow 2.3.0](https://www.tensorflow.org/)

## Abstract

Covariate shift occurs if the distribution of one or more of the covariates X in the test data significantly changes compared to X in the training data (Quionero-Candela et al., 2009)<a href="#references">[1]</a>. This can be very problematic, as the bias introudced by the covariate shift will have a negative effect on the generalization ability of the model. The goal of the paper then is to employ a monotonic lattice based model to mitigate the effect of covariate shift. This kind of model learns flexible monotonic functions by using calibrated interpolated look-up tables, the lattice (Gupta et al., 2016)<a href="#references">[2]</a>. The monotonicity constraint for individual features allows to model prior knowledge. As a first test of whether this approach is promising or not an experiment based on credit data is run. The goal here is to show that a TensorFlow Lattice (TFL) calibrated linear model (GAM) does not perform worse than a comparison classifier, in this case a Random Forrest model. 

## Getting started

For installing a local copy of the project just follow these simple implementation steps.

### Prerequisites

To run the code it is necessary to create a virtual environment for Python and TensorFlow. For doing so just run the following code:
  ```sh
  conda env create -f environment.yml
  ```
Alternatively `scr/03full_script.R` contains instructions how to set up a `virtualenv` environment.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Humboldt-WI/dissertations/covariate_shift.git
   ```
2. Go to project directory
   ```sh
   cd ~/path/to/project/directory/covariate_shift
   ```   
3. Install required R packages
   ```sh
   Rscript scr/helpers.R --install
   ```

## Preprocessing

To preprocess the data go to `scr` directory and run
   ```sh
   Rscript 01preprocess.R --dir --write=TRUE
   ```
This will preprocess the data and write it in `csv`format to `data/cleaned`. In case that there is no data available in `data/raw`, just run the above command with the `--web` option instead of `--dir`; this will automatically download all necessary data from corresponding web sources.

Furthermore there are additional options available to obtain EDA plots For more information see
   ```sh
   Rscript 01preprocess.R -h
   ```

## Training and evaluation

To train the model employed in the thesis run
   ```sh
   Rscript 02train.R --save
   ```
By default the model is only trained and evaluated on a single small data set and not on the entire data. To run the model on the full data, just suply the `-f` flag. The `--save` flag stores the training and test ROC values for the lattice and tree model in a data frame and writes it to `out/res`. Furthermore it is possible to change model parameters as batch size, epochs, etc. from the command line via corresponding flags and values. For default values call the help flag.

## Interactive script

Furthermore the repo contains the above scripts as an interactive `R`script `03full_script.R`, which allows for a deeper insight into the project code.

## References

 \[1\] Quionero-Candela, J., M. Sugiyama, A. Schwaighofer, and N. D. Lawrence (2009): Dataset shift in machine learning, The MIT Press.
 
\[2\] Gupta, M., A. Cotter, J. Pfeifer, K. Voevodski, K. Canini, A. Mangylov, W. Moczydlowski, and A. Van Esbroeck (2016): “Monotonic calibrated interpolated look-up tables,” The Journal of Machine Learning Research, 17, 3790–3836.

\[3\] Garcia, E. and M. Gupta (2009): “Lattice regression,” Advances in Neural Information Processing Systems, 22, 594–602.

<!-- ```
$ pwd
/path/to/project/directory/ba_vwl

$ ls
|- ba_vwl.Rproj
|- data/
   |- cleaned/
   |- raw/
	   |- german/
	   |- gmc/
	   |- pak/
	   |- taiwanese/
   |- README.md
|- environment.yml
|- out/
   |- res/
   |- plots/
|- README.md
|- scripts/
   |- feature_configs.py
   |- helpers.R
   |- tfl_experiment.R

``` -->