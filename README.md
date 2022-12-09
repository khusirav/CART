# CART
Classification and regression tree python realisation

## Installation (Windows)

Download the repo and open it in cmd.

Create python virtual environment using venv:
```
$ python -m venv cart_venv
```
Activate created environment:
```
$ cart_venv\scripts\activate
```
Install python libs from requirements.txt:
```
$ pip install -r requirements.txt
```
Run CART:
```
$ main.py
```

## Usage
To create and train CART starting with `tree_root`:
``` python
#dataset is tuple(np.ndarray of objects, np.ndarray of target classes)
tree_root = CART_create(dataset, tree_levels, classes_quantity, attr_steps)
```
To predict classes list using tree starting with `tree_root`:
``` python
classes_lst = CART_predict(dataset[0], tree_root)
```
To print the tree starting with `tree_root`:
``` python
CART_order(tree_root)
```
 
