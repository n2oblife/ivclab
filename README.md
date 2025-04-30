# IVC Lab Code Repository

Source code for the Image and Video Compression Lab 
lectures at the Technical University of Munich

## How to setup the codebase

Create a conda environment & activate
```bash
conda create --name ivclabenv
conda activate ivclabenv
```

Install necessary packages

```bash
pip install -r requirements.txt
pip install -e .
```

## How to run the test cases

```bash
python -m unittest -v tests/ch1.py
```

## How to run the exercises

```bash
python exercises/ch1/ex1.py
```

## Download data

Click [here](https://www.moodle.tum.de/mod/resource/view.php?id=3367764) to download the necessary data.
