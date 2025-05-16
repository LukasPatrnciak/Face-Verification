# B I O M E T R I A
# Lukas Patrnciak
# AIS ID: 92320
# xpatrnciak@stuba.sk


# KNIZNICE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

from deepface import DeepFace
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import image as mpimg


# KONSTANTY
TARGET_SIZE = (640, 640)
TRESHOLD = 0.75


# FUNKCIE
def show_image_pair(img1_path, img2_path, title):
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Obrázky {img1_path} alebo {img2_path} nie sú dostupné.")
        return

    img1 = mpimg.imread(img1_path)
    img2 = mpimg.imread(img2_path)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.axis('off')
    plt.title('Obrázok 1')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title('Obrázok 2')

    plt.suptitle(title)
    plt.show()

def load_image_gray(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Obrázok {img_path} neexistuje.")

    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

def detect_and_compute(img):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def match_keypoints(desc1, desc2):
    """FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    """
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []

    for m, n in matches:
        if m.distance < TRESHOLD * n.distance:
            good.append(m)

    return good

def plot_matches(img1, kp1, img2, kp2, matches, title):
    outImg = np.zeros((1, 1, 3), dtype=np.uint8)  # Namiesto 'None"

    match_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, outImg,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(12,6))
    plt.imshow(match_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

def extract_features_resnet(model, img_path, target_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_resnet(img)
    features = model.predict(img)

    return features.flatten()


# SPRACOVANIE DAT
csv_path = 'face_verification.csv'
dataframe = pd.read_csv(csv_path)

matching_pairs = dataframe['target'].sum()
different_pairs = len(dataframe) - matching_pairs

print(f"Zhodných párov: {matching_pairs}")
print(f"Rozdielnych párov: {different_pairs}")

sample_matching = dataframe[dataframe['target'] == 1].iloc[0]
sample_different = dataframe[dataframe['target'] == 0].iloc[0]

image1_match = sample_matching['image_1']
image2_match = sample_matching['image_2']

image1_diff = sample_different['image_1']
image2_diff = sample_different['image_2']

show_image_pair(image1_match, image2_match, "Zhodný pár (rovnaká osoba, iná fotka)")
show_image_pair(image1_diff, image2_diff, "Rozdielny pár (rôzne osoby)")

image1_match_gray = load_image_gray(image1_match)
image2_match_gray = load_image_gray(image2_match)
image1_diff_gray = load_image_gray(image1_diff)
image2_diff_gray = load_image_gray(image2_diff)
image_same_gray = image1_match_gray


# SIFT
sift = cv2.SIFT_create()

kp1_self, des1_self = detect_and_compute(image_same_gray)
kp2_self, des2_self = detect_and_compute(image_same_gray)
good_matches_self = match_keypoints(des1_self, des2_self)

print(f"Počet dobrých zhôd (rovnaký obrázok sám so sebou): {len(good_matches_self)}")
plot_matches(image_same_gray, kp1_self, image_same_gray, kp2_self, good_matches_self, "Rovnaký obrázok 2x – zhoda bodov")

kp1_same, des1_same = detect_and_compute(image1_match_gray)
kp2_same, des2_same = detect_and_compute(image2_match_gray)
good_matches_same = match_keypoints(des1_same, des2_same)

print(f"Počet dobrých zhôd (rovnaká osoba - iná fotka): {len(good_matches_same)}")
plot_matches(image1_match_gray, kp1_same, image2_match_gray, kp2_same, good_matches_same, "Rovnaká osoba - iná fotka - zhoda bodov")

kp1_diff, des1_diff = detect_and_compute(image1_diff_gray)
kp2_diff, des2_diff = detect_and_compute(image2_diff_gray)
good_matches_diff = match_keypoints(des1_diff, des2_diff)

print(f"Počet dobrých zhôd (rôzne osoby): {len(good_matches_diff)}")
plot_matches(image1_diff_gray, kp1_diff, image2_diff_gray, kp2_diff, good_matches_diff, "Rôzne osoby - zhoda bodov")


# NATRENOVANE MODELY
print("Načítavanie ResNet50 modelu...")
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
print("ResNet50 načítaný")

y_true = []
resnet_distances = []
arcface_distances = []

for idx, row in dataframe.iterrows():
    image1_path = row['image_1']
    image2_path = row['image_2']
    label = row['target']

    try:
        feat1_resnet = extract_features_resnet(resnet_model, image1_path, TARGET_SIZE)
        feat2_resnet = extract_features_resnet(resnet_model, image2_path, TARGET_SIZE)

        result_arc = DeepFace.verify(img1_path=image1_path,
                                 img2_path=image2_path,
                                 model_name="ArcFace",
                                 enforce_detection=False,
                                 detector_backend="skip")

        dist_resnet = np.linalg.norm(feat1_resnet - feat2_resnet) # np.sqrt(np.sum((feat1_resnet - feat2_resnet)**2)), moze tam byt aj 512 priznakov
        dist_arc = 1 - result_arc["distance"]

        resnet_distances.append(dist_resnet)
        arcface_distances.append(dist_arc)

        y_true.append(label)

    except Exception as e:
        print(f"Chyba pri spracovaní riadku {idx}: {e}")
        continue

y_true = np.array(y_true)
print("Extrakcia príznakov hotová.")


# RESNET
resnet_distances = np.array(resnet_distances)
fpr_resnet, tpr_resnet, thresholds_resnet = roc_curve(y_true, -resnet_distances)
roc_auc_resnet = auc(fpr_resnet, tpr_resnet)
optimal_idx_resnet = np.argmax(tpr_resnet - fpr_resnet)
optimal_threshold_resnet = -thresholds_resnet[optimal_idx_resnet]
print(f"RESNET50: \nAUC: {roc_auc_resnet:.4f}\nOptimálny treshhold (TPR - FPR): {optimal_threshold_resnet:.4f}")


# ARCFACE
arcface_distances = np.array(arcface_distances)
fpr_arc, tpr_arc, thresholds_arc = roc_curve(y_true, arcface_distances)
roc_auc_arc = auc(fpr_arc, tpr_arc)
optimal_idx_arc = np.argmax(tpr_arc - fpr_arc)
optimal_threshold_arc = thresholds_arc[optimal_idx_arc]
print(f"ARCFACE: \nAUC: {roc_auc_arc:.4f}\nOptimály treshhold (TPR - FPR): {optimal_threshold_arc:.4f}")


# VIZUALIZACIA
plt.figure(figsize=(10,7))
plt.plot(fpr_resnet, tpr_resnet, label=f'ResNet50 AUC = {roc_auc_resnet:.2f}')
plt.plot(fpr_arc, tpr_arc, label=f'ArcFace AUC = {roc_auc_arc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC krivka pre verifikáciu tvárí')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ANALYZA NAJLEPSICH A NAJHORSICH PAROV
print("\nAnalyzujem extrémne prípady pomocou ArcFace...")

arcface_distances = np.array(arcface_distances)
y_true = np.array(y_true)

# TRUE pary
true_indices = np.where(y_true == 1)[0]
false_indices = np.where(y_true == 0)[0]

# TRUE: najmensie (dobre) a najvacsie (zle) vzdialenosti
true_sorted = sorted(true_indices, key=lambda j: arcface_distances[j], reverse=True)
false_sorted = sorted(false_indices, key=lambda j: arcface_distances[j], reverse=True)

print("\n TRUE páry – Najlepšie (min vzdialenosť):")
for i in true_sorted[:3]:
    row = dataframe.iloc[i]
    print(f"Dist: {arcface_distances[i]:.4f}")
    show_image_pair(row['image_1'], row['image_2'], f"TRUE (najlepšie - min. vzdialenosť): Vzdialenosť {arcface_distances[i]:.4f}")

print("\n TRUE páry – Najhoršie (max vzdialenosť):")
for i in true_sorted[-3:]:
    row = dataframe.iloc[i]
    print(f"Dist: {arcface_distances[i]:.4f}")
    show_image_pair(row['image_1'], row['image_2'], f"TRUE (najhoršie - max. vzdialenosť): Vzdialenosť {arcface_distances[i]:.4f}")

print("\n FALSE páry – Najmenšie vzdialenosti (potenciálne falošné zhody):")
for i in false_sorted[:3]:
    row = dataframe.iloc[i]
    print(f"Dist: {arcface_distances[i]:.4f}")
    show_image_pair(row['image_1'], row['image_2'], f"FALSE (potenciálna falošná zhoda - min. vzdialenosť): {arcface_distances[i]:.4f}")

print("\n FALSE páry – Najväčšie vzdialenosti (očakávané správne):")
for i in false_sorted[-3:]:
    row = dataframe.iloc[i]
    print(f"Dist: {arcface_distances[i]:.4f}")
    show_image_pair(row['image_1'], row['image_2'], f"FALSE (potenciálne dobré rozlíšenie - max. vzdialenosť): {arcface_distances[i]:.4f}")


# TROJICKOVA VERIFIKACIA
print("\nTrojičková verifikácia pomocou ArcFace (DeepFace)...")

triplet_data = []
true_pairs = dataframe[dataframe['target'] == 1].reset_index(drop=True)
false_pairs = dataframe[dataframe['target'] == 0].reset_index(drop=True)

for i in range(min(len(true_pairs), len(false_pairs))):
    anchor = true_pairs.loc[i, 'image_1']
    positive = true_pairs.loc[i, 'image_2']
    negative = false_pairs.loc[i, 'image_2']
    triplet_data.append((anchor, positive, 1))
    triplet_data.append((anchor, negative, 0))

triplet_dataframe = pd.DataFrame(triplet_data, columns=['image_1', 'image_2', 'target'])
triplet_dataframe.to_csv("triplet_face_verification.csv", index=False)
print("Upravený CSV súbor bol uložený ako 'triplet_face_verification.csv'")

arcface_triplet_distances = []
y_true_triplets = []

for idx, row in triplet_dataframe.iterrows():
    try:
        result = DeepFace.verify(img1_path=row['image_1'],
                                 img2_path=row['image_2'],
                                 model_name="ArcFace",
                                 enforce_detection=False,
                                 detector_backend="skip")

        distance = 1 - result["distance"]
        (arcface_triplet_distances.append(distance))
        y_true_triplets.append(row['target'])

    except Exception as e:
        print(f"Chyba pri spracovaní riadku {idx}: {e}")
        continue

# ROC a AUC
arcface_triplet_distances = np.array(arcface_triplet_distances)
y_true_triplets = np.array(y_true_triplets)

fpr, tpr, thresholds = roc_curve(y_true_triplets, arcface_triplet_distances)
roc_auc = auc(fpr, tpr)

# Optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"ARCFACE TRIPLET: \nAUC: {roc_auc:.4f}\nOptimálny treshhold (TPR - FPR): {optimal_threshold:.4f}")


# ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Triplet ArcFace AUC = {roc_auc:.4f}')
plt.plot(fpr_arc, tpr_arc, label=f'ArcFace AUC = {roc_auc_arc:.4f}')
plt.plot(fpr_resnet, tpr_resnet, label=f'ResNet50 AUC = {roc_auc_arc:.4f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Krivka')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# Konfuzna matica
predictions_triplet = (arcface_triplet_distances >= optimal_threshold).astype(int)
cm_acrface_triplet = confusion_matrix(y_true_triplets, predictions_triplet)
ConfusionMatrixDisplay(cm_acrface_triplet, display_labels=["False", "True"]).plot()
plt.title("Konfúzna matica - Trojičková verifikácia")
plt.grid(False)
plt.show()

predictions_acrface = (arcface_distances >= optimal_threshold_arc).astype(int)
cm_acrface = confusion_matrix(y_true, predictions_acrface)
ConfusionMatrixDisplay(cm_acrface, display_labels=["False", "True"]).plot()
plt.title("Konfúzna matica - AcrFace verifikácia")
plt.grid(False)
plt.show()

predictions_resnet50 = (resnet_distances >= optimal_threshold_resnet).astype(int)
cm_resnet50 = confusion_matrix(y_true, predictions_resnet50)
ConfusionMatrixDisplay(cm_resnet50, display_labels=["False", "True"]).plot()
plt.title("Konfúzna matica - ResNet50 verifikácia")
plt.grid(False)
plt.show()