# Rule of thumb and $N_{{eff}}$ code
## Prerequisite files
You'll need to download the model files from the NANOGrav 15yr New Physics analyses from https://zenodo.org/records/8092761.

Then, you can apply my patch to make these modules JAX compatible. Assuming you download and unpack the model files to `models_1.0.1`, use the following command:

``` sh
patch --directory models_1.0.1 -p1 < models_patch.diff
```
## Python environment
You can use the `environment.yml` file to recreate the python environment you'll need to run the code.

``` sh
conda env create -f environment.yml
```

## Run the code
The `models_1.0.1` directory you downloaded should be renamed to `models`. It also needs to be in the same directory as the script/notebook.
You can run the scripts or use the notebooks; the scripts are just automatic conversions of the notebooks from `nbconvert`.
