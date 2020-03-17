# Spatio-temporal data recovery module

## Installation

1. Install GHER DINEOF: http://modb.oce.ulg.ac.be/mediawiki/index.php/DINEOF  
After this step you should have shell command calling GHER DINEOF

2. Install dependencies  
```bash
pip install -r requirements.txt  # Python 3 is required
```

## Usage

1. Collect data  
It should have **\*[A-Z]\*YYYYDDD.\*.nc** format and these groups:
    * **navigation_data**  

            With variables:  

                1. longitude  
                2. latitude

    * **geophysical_data**  

            With variables:  

                1. investigated object which you will specify in data_desc_example.yaml

2. Put meta information in **data_desc_example.yaml** (There are good comments, do not hesitate to check it)

3. Run this package in your python 3.* scripts  
```python
# Example
from dineof.model import Dineof

d = Dineof('PUT PATH TO data_desc_example.yaml HERE')

d.fit()  # By default it keeps data only for summer
d.predict()

```
