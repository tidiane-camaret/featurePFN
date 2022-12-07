from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from featurePFN.transformer_pred_interface import TabPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
# When N_ensemble_configurations > #features * #classes, no further averaging is applied.

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=1, no_preprocess_mode=True, only_inference=False)
X_train = Variable(torch.from_numpy(X_train).float(), requires_grad=True)
y_train = torch.from_numpy(y_train).float()
classifier.fit(X_train, y_train)
X_test= torch.from_numpy(X_test).float()
p = classifier.predict(X_test, return_winning_probability=True)

print(p.sum().backward())
