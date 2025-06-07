# Ruta base del proyecto
PYTHONPATH := .

# Ejecutar tests con pytest
test:
	PYTHONPATH=$(PYTHONPATH) pytest -v tests/

# Formatear código con black
format:
	black tree_algorithms/ tests/ main.py

# Limpiar archivos temporales
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -r {} +

# Reinstalar entorno desde cero
install:
	pip install -r requirements.txt
