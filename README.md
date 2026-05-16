# Lung Cancer Classification Project

This project provides deep learning models for classifying lung cancer using CT scan and pathology images. The models are implemented in Jupyter Notebooks and leverage state-of-the-art architectures.

## 📁 Notebooks Included

There are **four** Jupyter notebooks in this repository:

1. **train_transfomer_deit - ct scan.ipynb**
    - Uses the DeiT Transformer for CT scan image classification.
2. **train_transfomer_deit - pathalogy.ipynb**
    - Uses the DeiT Transformer for pathology image classification.
3. **train-EfficientNet-ct-scan.ipynb**
    - Uses EfficientNet for CT scan image classification.
4. **train-resnet-ct-scan.ipynb**
    - Uses ResNet for CT scan image classification.

## 🏃‍♂️ How to Run the Notebooks

1. **Install Requirements**  
   Make sure you have Python 3.8+ and Jupyter installed. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. **Open a Notebook**  
   Launch Jupyter Notebook or JupyterLab:

    ```bash
    jupyter notebook
    ```

    Then open any of the four notebooks listed above.

3. **Run All Cells**  
   In the notebook interface, select `Cell` > `Run All` to execute all code cells. This will train and/or test the model as described in each notebook.

## 📂 Datasets

- CT scan images: `dataset/ct-scan/{aca, lcc, norm, scc}`
- Pathology images: `dataset/pathalogy/{aca, norm, scc}`

## 📝 Notes

- Each notebook is self-contained and includes instructions and explanations.
- You can modify dataset paths or parameters as needed for your experiments.

---

For any questions, please refer to the notebook markdown cells or contact the project maintainer.
