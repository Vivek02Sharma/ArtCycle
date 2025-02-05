# ArtCycle🎨

ArtCycle is a deep learning project that leverages the power of CycleGAN (Generative Adversarial Networks) to transform photos into paintings and vice versa. The model works in an unsupervised manner, enabling image-to-image translation without paired data. With ArtCycle, you can upload a photo and see it converted into an artistic painting or take a painting and turn it into a photo.
[Visit the ArtCycle](https://artcycle-ai.streamlit.app/)

![cyclegan-architecture](https://github.com/user-attachments/assets/b1152d4b-923a-473a-a4ff-1b631e89bd79)
<p align="center"><a href="https://arxiv.org/pdf/1703.10593">CycleGAN research paper</a></p>

## 📂 Directory structure:

```
ArtCycle/
    ├── download_dataset.sh
    ├── loss.py
    ├── main.py
    ├── model.py
    ├── requirements.txt
    ├── train.py
    ├── assets/
    └── .streamlit/
        └── config.toml   
```

## 🧑‍💻 How to Train:

**1. Clone the repository:**

```
https://github.com/Vivek02Sharma/ArtCycle.git
```

**2. Install dependencies:**

```
pip install -r requirements.txt
```

**3. Download the dataset:**

```
bash download_dataset.sh
```

**4. Training the Model:**
```
python train.py
```

## 🚀 How to use:

- **Run the application**
```
streamlit run main.py
```

## 📓 Kaggle Notebook link:
[CycleGan-photo2painting](https://www.kaggle.com/code/viveksharmar2/cyclegan-image2painting)
