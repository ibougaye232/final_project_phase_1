import numpy as np

# Charger le fichier npz
data = np.load("images_dataset.npz")

# Accéder aux images et aux étiquettes
images = data['images']
labels = data['labels']
print(images)
print(labels)

# Maintenant, vous pouvez utiliser 'images' et 'labels' comme bon vous semble
