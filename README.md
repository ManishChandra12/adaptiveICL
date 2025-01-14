# Adaptive-ICL

## Setup
1. Clone the repo
```
git clone https://github.com/ManishChandra12/adaptiveICL.git
```
2. cd into the project directory
```
cd adaptiveICL
```
3. Create the virtual environment
```
conda env create aicl -f environment.yml
```
4. Activate the virtual environment
```
conda activate aicl
```
5. Run the following to get information on all the available arguments
```
python -m src.main --help
```

## Training and Evaluation
6. Example command to run FICL baseline
```
python -m src.main --dataset sst2 --method static --model_name microsoft/phi-2 --static_split dev --K_max 2
```
Once the best performing `k` is identified using dev set, modify `dev` to `test` to run FICL on the test set with appropriate `--K_max`.\
7. Example command to generate data for training the k-predictor model
```
python -m src.main --prepare --dataset sst2 --method dynamic --model_name microsoft/phi-2 --oracle_split train
```
Modify `train` to `test` to run oracle on the test set.\
8. Example command to run AICL
```
python -m src.main --dataset sst2 --method dynamic --model_name microsoft/phi-2
```
