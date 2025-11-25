Install: Python 3.13.7
Make an environment: python3 -m venv env1lab
Go into environment: source ~/env1lab/bin/activate
Install packages: pip install pandas numpy pillow psutil
Use script: python apitopng.py (runs turning API csv to PNG)

Make an environment: python3 -m venv ~/env2lab
Go into environment: source ~/env2lab/bin/activate
Install packages: pip install numpy pandas pillow scikit-learn xgboost psutil
Place kaggle Dataset folder in the relative path from the python script: ‘./Dataset’
Place Virus‑MNIST Dataset folder in the relative path from the python script: ‘./MNIST’
Use script: python 2Train.py (this uses kaggle dataset)
Use script: python 18Train.py (this uses Virus‑MNIST dataset)