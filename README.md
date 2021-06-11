## Domain-adaptive Neural Networks For Cross-Species TF Binding Prediction

This repository is the codebase for reproducing all of the analyses from Cochran et al. 2021, "Domain adaptive neural networks improve cross-species prediction of transcription factor binding" (preprint at https://www.biorxiv.org/content/10.1101/2021.02.13.431115v1). Some code is adapted from work by @DivyanshiSrivastava, and the gradient reversal layer implementation is modified from @michetonu (Michele Tonutti) at https://github.com/michetonu/gradient_reversal_keras_tf ; otherwise the code here is authored by KC.

In this project, we trained neural networks to predict transcription factor binding in one species (mouse), and then assessed and improved upon the performance of these networks in a previously unseen species (human). The model architecture used is a hybrid convolutional-LSTM neural network. We trained both a standard (non-domain-adaptive) architecture and a domain-adaptive architecture, augmented with a second training task head and a gradient reversal layer, and showed that the domain-adaptive approach can reduce false-positive predictions on species-unique repeat elements in the cross-species prediction scenario. This codebase contains all the scripts needed to reproduce the data preprocessing, model training, and downstream analyses.


### Quick-access Roadmap: Why Are You Here?

**1. I want to learn how to implement a gradient reversal layer.**

See `2_train_and_test_models/flipGradientTF.py` for the implementation, `2_train_and_test_models/DA_train.py` for its use in model architecture setup.

**2. I want to know how you ran peak calling or what settings were used.**

Peak calling was performed using MultiGPS. Details on how to run MultiGPS can be found at http://mahonylab.org/software/multigps. See `0_preprocess_genome_and_peaks/peak_calling/run_multiGPS.sh` on the specific input arguments given to MultiGPS here. Note that this script requires use of a READDB instance, read distribution file, and design file for each peak call run.  This script is currently out-of-date with the rest of the repo because its filepaths are specific to the Mahony Lab's MultiGPS setup on Penn State's ACI computing cluster.

**3. I want to know where genome annotation tracks (Umap, ENCODE exclusion list, RepeatMasker) were downloaded from. **

See `setup_directories_and_download_files.sh` for exact URLs. See the ENCODE exclusion list paper (Amemiya et al. 2019, https://www.nature.com/articles/s41598-019-45839-z), the Umap/Bismap site by the Hoffman Lab (https://bismap.hoffmanlab.org/), and http://repeatmasker.org/ or http://genome.ucsc.edu/cgi-bin/hgTrackUi?db=hg38&g=rmsk for more information on how these tracks were created.

**4. I want to know the exact model architecture you used.**

See `2_train_and_test_models/train.py` for the basic (non-domain-adaptive) model architecture implementation and `2_train_and_test_models/DA_train.py` for the domain-adaptive model. These two scripts rely on hyperparameters that are stored in classes in `2_train_and_test_models/default_params.py` and `2_train_and_test_models/DA_params.py`, respectively.

**5. I want to know how you pre-processed your datasets after peak calling.**

The directory `0_preprocess_genome_and_peaks/` contains all of the scripts responsible for generating "windows" from the genome, converting peak calls into binary labels for those windows, and filtering the genome-wide window set by the ENCODE exclusion list and Umap coverage. The directory `1_make_training_and_testing_data/` contains all of the scripts responsible for converting the full genome's worth of "windows" into separated training, validation, and test sets, with specific file setups required by the models' data loaders. Both directories contain a "runall" file showing the order you would run the scripts within the directory.

**6. I want to reproduce your entire workflow from scratch.**

See the section below!


## Full Workflow

### Installation

First, make a directory for everything to happen in, and cd into that directory. Then download this repository and cd into it.

```
mkdir -p "/users/me/domain_adaptation"
cd "/users/me/domain_adaptation"
git clone http://git@github.com/seqcode/cross-species-domain-adaptation.git
cd cross-species-domain-adaptation
```

The script `setup_directories_and_download_files.sh` will build the directory structure for all the data and will download all genome annotation files needed, besides the FASTAs (download these yourself if you don't have them, e.g. from https://www.gencodegenes.org/). You just need to pass in the project root you created in the last step.

```
./setup_directories_and_download_files.sh "/users/me/domain_adaptation"
```

Now that a data directory structure exists, you will need to place your called peaks bed files in the correct spots. The result will be the raw_data dir inside the project directory looking like this:

```
└── raw_data
    ├── hg38
    │   ├── CEBPA
    │   │   └── mgps_out_CEBPA.bed
    │   ├── CTCF
    │   │   └── mgps_out_CTCF.bed
    │   ├── Hnf4a
    │   │   └── mgps_out_Hnf4a.bed
    │   └── RXRA
    │       └── mgps_out_RXRA.bed
    └── mm10
        ├── CEBPA
        │   └── mgps_out_CEBPA.bed
        ├── CTCF
        │   └── mgps_out_CTCF.bed
        ├── Hnf4a
        │   └── mgps_out_Hnf4a.bed
        └── RXRA
            └── mgps_out_RXRA.bed
```

Where each bed file is the peak calls for a given TF in a given species. The required format for these files is that there are (at least) three tab-separated columns with chromosome, start coordinate, and stop coordinate listed in standard BED format. The coordinate should correspond to the ~summit~, or center of the peak if you don't have summits (meaning, the number in column 3 should be the number in column 2 plus 1). Feel free to modify `0_preprocess_genome_and_peaks/_make_tf_labels.sh` if you want to accomodate a different peak calling file format -- this script will be looking for the exact file names above so be careful there.

Lastly, see inside `0_preprocess_genome_and_peaks/0_runall_preprocess_data.sh` where paths to genome FASTA files are written -- modify these paths to point to your genome FASTAs. Then you can run all the data pre-processing code, passing in the main directory path each time.

```
source 0_preprocess_genome_and_peaks/0_runall_preprocess_data.sh "/users/me/domain_adaptation"
source 1_make_training_and_testing_data/1_runall_setup_model_data.sh "/users/me/domain_adaptation"
```

Now you are ready to train models!

Direct the python scripts to the correct filepaths by editing the "ROOT" and the two genome filepaths in `2_train_and_test_models/default_params.py`. Then you can run either `run_training.sh` to train non-domain-adaptive models or `run_DA_training.sh` to train domain-adaptive models. Afterwards, you can walk through any downstream analysis or re-create any of the main figures from the manuscript using the Jupyter notebooks in `3_manuscript_figure_and_table_notebooks/`. That's it!

## Dependencies
- Python ~ 3.7
- Keras: 2.3
- tensorflow-base, tensorflow-gpu: 2.1.0
- numpy: 1.19
- seqdataloader: 0.130
  - pybigwig: 0.3.17
- pyfaidx: 0.5.9 (only to make FASTA index)
- bedtools: 2.26
- scikit-learn: 0.23
- scipy: 1.5
- pandas: 1.0 (jupyter notebooks only)
- Jupyter/IPython and nb_conda_kernels (jupyter notebooks only)
