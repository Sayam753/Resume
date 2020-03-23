For making the documentation, create a new virtual environment using `mkvirtualenv` command.
Install the requirements -

```bash
pip install sphinx sphinx_rtd_theme
```

Create a new directory `docs` to store documentation and `cd` into it.
Run the command `sphinx-quickstart` and enter
1. Separate source and build directories (y/n) [n]: n
2. Project name: OSINT-SPY
3. Author name(s): Isha, Jayashree, Nethra, Pradum, Sushant, Sayam
4. Project release []: 1.0.0
5. Project language [en]: en

Open file `conf.py` and change -

```python
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
```

to 

```python
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
```

Add a string to `extensions` list as

```python
extensions = ['sphinx.ext.autodoc'
]
```

Change the theme to `sphinx_rtd_theme` - 

```python
html_theme = 'sphinx_rtd_theme'
```

Close and save the file. Open file `index.rst` and `modules` name below toctree. This should look something like this - 

```bash
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
```



Move to previous directory and create a python file(any name) and add any function(having documentation).

```python
def test_function():
	"""
	This is testing function to check if documentation is made properly or not
	We will populate all sections of actual code later.
	"""
```

Move to docs, run the following command -

```bash
sphinx-apidoc -o . ..
make html
open _build/html/index.html
```
