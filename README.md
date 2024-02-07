### Regional and global brain age prediction model
---
- The code in this repository pertains to brain age models that are based on the cortical surface. 
- The model was constructed and trained by Gilsoon Park.
- Convolutional graphical neural networks were used (https://github.com/mdeff/cnn_graph).
</br>
### Dependencies
---
* OS: CentOS Linux release 7.9.2009 (Core)
* python==3.7.10
* tensorflow==1.15.0
* pandas==1.3.5
* numpy==1.20.2
* scipy==1.6.2
* sklearn=0.24.2

### Installation
---
1. Clone this repository.
   ```sh
   git clone https://github.com/pks1207/regional_Brain_age
   cd regional_Brain_age
   ```

2. Install the dependencies. The code should run with TensorFlow 1.x. 

3. Modify the Python code (Regional_and_global_brain_age_prediction_model.py) to use user data

### Usage
---

1. Extract cortical thickness and gray matter/white matter intensity ratio by using the CIVET pipeline (https://www.bic.mni.mcgill.ca/ServicesSoftware/CIVET-2-1-0-Table-of-Contents) from target subjects.

2. Create feature text files like the files in the test_dataset folder in this repository. The first column is cortical thickness, and the second column is gray matter/white matter intensity ratio.

3. Modify the Python code (Regional_and_global_brain_age_prediction_model.py) by modifying the test_list variable as shown in line 26 of the code.

4. The Regional_and_global_brain_ages.csv file is then created.

5. Perform delta correction by using the brain ages of your normal subjects.
