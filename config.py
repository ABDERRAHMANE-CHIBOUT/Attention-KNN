# config.py

class Config:
    # ---- Shared ----
    K       = 7
    lr      = 0.001
    epochs  = 30

    # ---- Image Classification (CIFAR-10) ----
    image_max_samples  = 5000
    image_pca_components = 128    # input_dim for image model
    image_num_classes  = 10

    # ---- Sentiment Analysis ----
    text_max_features  = 1000     # input_dim for text model
    text_num_classes   = 2

