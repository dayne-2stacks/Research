import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from scipy.spatial import Delaunay
from torch_geometric.utils import to_networkx
from visualize import visualize_graph
from gcn import GCN
from karateclub import Graph2Vec
import random
from collections import defaultdict

# Define batch size
batch_size = 5

counter = Counter()


class GraphDataset(InMemoryDataset):
    def __init__(self, root, fileDict, transform=None, pre_transform=None):
        self.fileDict = fileDict
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_names = []
        for set_type, folders in self.fileDict.items():
            for folder in folders:
                folder_path = os.path.join(self.root, folder)
                file_names.extend([f for f in os.listdir(folder_path) if f.endswith('.tsv')])
        return file_names

    @property
    def processed_file_names(self):
        return 'data.pt'


    def download(self):
        pass

    def process(self):
        graphs = []
        labels = []
        data_list = []
        idx = 0
        for set_type, folders in self.fileDict.items():
            annotation_file = os.path.join(self.root, f"{set_type}.csv")
            annotation_df = pd.read_csv(annotation_file)

            for folder in folders:
                folder_path = os.path.join(self.root, folder)

                for filename in os.listdir(folder_path):
                    if filename.endswith('.tsv'):
                        tsv_path = os.path.join(folder_path, filename)
                        data_df = pd.read_csv(tsv_path, sep='\t')

                        # Extract annotations
                        graph_annotation = annotation_df.iloc[idx].values
                        graph_annotation = pd.to_numeric(graph_annotation, errors='coerce')
                        graph_annotation = graph_annotation[~np.isnan(graph_annotation)]
                        graph_annotation = torch.tensor(graph_annotation, dtype=torch.long)

                        # Node positions
                        x = torch.tensor(data_df[['x', 'y']].values, dtype=torch.float)

                        # Delaunay triangulation
                        tri = Delaunay(data_df[['x', 'y']].values)
                        triangles = torch.tensor(tri.simplices, dtype=torch.long)
                        edges = []
                        for tri in triangles.tolist():
                            edges.append([tri[0], tri[1]])
                            edges.append([tri[1], tri[2]])
                            edges.append([tri[2], tri[0]])
                        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

                        # Create Data object
                        data = Data(x=x, edge_index=edge_index, y=graph_annotation)

                        # Convert to NetworkX graph
                        G = to_networkx(data, to_undirected=True)

                        # Assign node labels based on degree
                        for node in G.nodes():
                            G.nodes[node]['label'] = str(G.degree[node])

                        graphs.append(G)
                        labels.append(graph_annotation.item())
                        data_list.append(data)
                        idx += 1

        # Balance the dataset
        data_list = self.balance_dataset(data_list, labels)

        # Compute graph embeddings on balanced data
        graphs_balanced = [to_networkx(data, to_undirected=True) for data in data_list]
        for G in graphs_balanced:
            for node in G.nodes():
                G.nodes[node]['label'] = str(G.degree[node])

        model = Graph2Vec(dimensions=64)
        model.fit(graphs_balanced)
        embeddings = model.get_embedding()

        # Add embeddings to Data objects
        for data_obj, emb in zip(data_list, embeddings):
            data_obj.graph_embedding = torch.tensor(emb, dtype=torch.float).unsqueeze(0)

        # Save the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def balance_dataset(self, data_list, labels):
        from collections import defaultdict
        import random
        # Organize data by class
        class_to_data = defaultdict(list)
        for data, label in zip(data_list, labels):
            class_to_data[label].append(data)

        # Find the maximum class count
        max_count = max(len(v) for v in class_to_data.values())

        # Balance the dataset
        balanced_data_list = []
        for class_label, data_samples in class_to_data.items():
            num_to_add = max_count - len(data_samples)
            if num_to_add > 0:
                # Apply data augmentation
                augmented_data = [self.augment_data(random.choice(data_samples)) for _ in range(num_to_add)]
                data_samples.extend(augmented_data)
            balanced_data_list.extend(data_samples)

        random.shuffle(balanced_data_list)
        return balanced_data_list

    def augment_data(self, data):
        import copy
        data_aug = copy.deepcopy(data)
        # add small noise to node features
        noise = torch.randn_like(data_aug.x) * 0.01
        data_aug.x += noise
        return data_aug


    # def len(self):
    #     return len(self.processed_file_names)

    # def get(self, idx):
    #     return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    

def print_graph_and_visualize(data):
    print()
    print(data)
    print(data.y)
    # Create a NetworkX graph from PyTorch Geometric Data object
    G = to_networkx(data, to_undirected=True)
    
    # Visualize the graph
    pos = {i: data.x[i].numpy() for i in range(data.x.shape[0])}

    # Visualize the graph with actual (x, y) positions
    visualize_graph(G, pos, color=torch.fill(torch.zeros(data.x.shape[0]), data.y[0]))

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch, data.graph_embedding)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.



if __name__ == "__main__":


    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    # Assuming 'classes' is defined as:
    classes = ('plain_arch', 'right_loop', 'left_loop', 'tented_arch', 'whorl')

    
    trainData = GraphDataset('', {"training": ["R1", "R2", "R3"]})
    testData = GraphDataset('', {"testing": ["R4"]})
    validateData = GraphDataset('', {"validation": ["R5"]})

    # class_to_data = defaultdict(list)

    # # Collect all labels from the training dataset
    # all_labels = []
    # for data in trainData:
    #     all_labels.append(data.y.item())
    #     class_to_data[data.y.item()].append(data)

    # # Compute class weights
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(classes)), y=all_labels)

    # # Convert class weights to a tensor
    # class_weights = torch.tensor(class_weights, dtype=torch.float)


    # max_count = max(len(v) for v in class_to_data.values())

    # balanced_data_list = []
    # for class_label, data_list in class_to_data.items():
    #     num_to_add = max_count - len(data_list)
    #     if num_to_add > 0:
    #         data_list.extend(random.choices(data_list, k=num_to_add))
    #     balanced_data_list.extend(data_list)



    train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=False)
    validation_dataloader = DataLoader(validateData, batch_size=batch_size, shuffle=False)



    # print_graph_and_visualize(data)
    for count in counter:
        print(count, counter[count])

    data = trainData[0]

    # print_graph_and_visualize(data)


    # Define model
    model = GCN(hidden_channels=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Define the loss function with class weights
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = torch.nn.CrossEntropyLoss()


    



    PATH = './fingerprint.pth'
    best_val_loss = float('inf')  # initialize the best validation loss
    patience = 3  # number of epochs to wait before early stopping if no improvement
    epochs_no_improve = 0  # counter for early stopping

    for epoch in range(200):
        model.train()
        running_loss = 0.0 
        for i, data in enumerate(train_dataloader, 0):  # Iterate in batches over the training dataset.
            optimizer.zero_grad()  # Clear gradients.
            out = model(data.x, data.edge_index, data.batch, data.graph_embedding)  # Perform a single forward pass 
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

            # accumulate training loss
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


       
        model.eval()  # switch to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # no gradient calculation for validation
            for data in validation_dataloader:
                # Update to work with PyTorch Geometric data structure
                out = model(data.x, data.edge_index, data.batch, data.graph_embedding)
                loss = criterion(out, data.y)
                val_loss += loss.item()

        val_loss /= len(validation_dataloader)
        print(f'Epoch {epoch+1} Validation Loss: {val_loss:.3f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), PATH)
            epochs_no_improve = 0  # reset counter if validation loss improves
        else:
            epochs_no_improve += 1  # increment if no improvement
            if epochs_no_improve >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break

    # testAcc = test(test_dataloader)
    # print(f'The accuracy is {testAcc}')
    classes = ('plain_arch', 'right_loop', 'left_loop', 'tented_arch', 'whorl')
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            labels = data.y
            outputs = model(data.x, data.edge_index, data.batch, data.graph_embedding)  
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')


        # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_dataloader:
            labels = data.y
            outputs = model(data.x, data.edge_index, data.batch, data.graph_embedding)  
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
