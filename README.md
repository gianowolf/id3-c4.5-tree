# ID3 and C4.5 Decision Tree Implementation

This project provides a modular implementation of the ID3 and C4.5 decision tree algorithms for supervised classification tasks.

## Features
- Entropy, Information Gain, and Gain Ratio
- Support for categorical and numerical attributes
- Clean modular design for reuse and testing

## Project Structure
- `tree_algorithms/`: core modules implementing ID3 and C4.5
- `main.py`: sample usage
- `data/`: test datasets

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

## Running

```bash
python main.py
```

## License

MIT License

````

---

### `requirements.txt`
```txt
python-dotenv
pandas
````
---

### `main.py`

```python
from tree_algorithms import id3, c45

# Placeholder para ejemplo de uso con dataset
if __name__ == "__main__":
    print("Decision Tree Project (ID3 / C4.5)")
```

---

### `tree_algorithms/__init__.py`

```python
# Empty init for package
```
