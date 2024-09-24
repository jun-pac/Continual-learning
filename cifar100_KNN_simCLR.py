
# Memory constrainted ER


from os.path import expanduser
from annoy import AnnoyIndex
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np

import argparse
import torch
from torchvision.transforms import ToTensor
import torch.optim.lr_scheduler
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.benchmarks import nc_benchmark
from networks import *

from avalanche.training.supervised import Naive
from avalanche.training.plugins import EWCPlugin, ReplayPlugin, GEMPlugin, LwFPlugin

from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def extract_features(dataloader, model):
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.cuda() 
            outputs = model(inputs)
            features.append(outputs.view(outputs.size(0), -1).cpu().numpy())
            labels.append(targets.cpu().numpy())
    
    return np.concatenate(features), np.concatenate(labels)





def main(args):
    # Model getter: specify dataset and depth of the network.
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
    resnet50.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    
    device = torch.device(
    # --- CONFIG
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    benchmark = SplitCIFAR100(
        n_experiences=10,
        fixed_class_order=[99-i for i in range(100)]
    ) 
    # We woudln't use first_exp_with_half_classes option


    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # cl_strategy = JointTraining( 
    #     model,
    #     torch.optim.SGD(model.parameters(), lr=0.01),
    #     CrossEntropyLoss(),
    #     train_mb_size=50,
    #     train_epochs=50, # To see final score, Infinetly many epoch is not 
    #     eval_mb_size=50,
    #     device=device,
    #     evaluator=eval_plugin,
    # )


    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for i,experience in enumerate(benchmark.train_stream):
        print("Start of experience ", experience.current_experience)
        # cl_strategy.train(experience)

        print(f"i : {i} / type(experience)  {type(experience)}")
        # results.append(cl_strategy.eval(benchmark.test_stream))

        # train_features, train_labels = extract_features(trainloader, resnet50)
        # d = train_features.shape[1] 
        # annoy_index = AnnoyIndex(d, 'euclidean')  
        # for i in range(train_features.shape[0]):
        #     annoy_index.add_item(i, train_features[i])
        # annoy_index.build(10) # Rebuild everytime whenever new data is added????
        # test_features, test_labels = extract_features(testloader, resnet50)


        # k = 5 
        # predictions = []
        # for test_feature in test_features:
        #     neighbors = annoy_index.get_nns_by_vector(test_feature, k, include_distances=False)
        #     neighbor_labels = train_labels[neighbors]
        #     predictions.append(np.bincount(neighbor_labels).argmax())
        
        # accuracy = np.mean(np.array(predictions) == test_labels)

        # D, I = index.search(test_features.astype('float32'), k)
        # predictions = []
        # for neighbors in I:
        #     neighbor_labels = train_labels[neighbors]
        #     predictions.append(np.bincount(neighbor_labels).argmax())
        # predictions=np.array(predictions)
        # accuracy = np.mean(predictions == test_labels)

        # How about evaluation datas?



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)

# conda activate vv
# python cifar100_ER.py | tee log/ER_2000
# python cifar100_ER.py | tee log/ER_5000
# python cifar100_ER.py | tee log/ER_10000
# python cifar100_ER.py | tee log/ER_20000

