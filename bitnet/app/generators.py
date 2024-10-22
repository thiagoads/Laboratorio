from sklearn.datasets import make_blobs
import numpy as np


class LinearlySeparableDataGenerator():
    def __init__(self):
        pass

    def generate(self,
        num_samples:int, 
        num_classes:int,
        num_features:int,
        random_state: int):  
        X, y = make_blobs(
                       n_samples = num_samples, 
                       n_features=num_features,
                       centers=num_classes,
                       cluster_std=2.0,
                       random_state=random_state, 
                       shuffle=True
                       )
        return X, y
    

class NonLinearlySeparableDataGenerator():

    def __init__(self):
        pass

 
    def generate(self, 
                 num_samples: int, 
                 num_classes:int,
                 num_features:int,
                 random_state: int,
                 noise=0.05, 
                 factor:int=0.5):
        
        np.random.seed(random_state)
        
        X = []
        y = []

        samples_per_class = num_samples // num_classes
        remainder = num_samples % num_classes

        # Distribute the remainder to one of the classes randomly or to the first if reproducible
        extra_samples = [1 if i < remainder else 0 for i in range(num_classes)]
        
        # Generate each class as a circle with increasing radius
        for i in range(num_classes):
            # Adjust the number of samples for this class considering the remainder
            adjusted_num_samples = samples_per_class + extra_samples[i]
            radius = factor * (i + 1)  # Incremental radius for each class
            theta = np.linspace(0, 2 * np.pi, adjusted_num_samples)
            x_circle = radius * np.cos(theta) + np.random.normal(scale=noise, size=adjusted_num_samples)
            y_circle = radius * np.sin(theta) + np.random.normal(scale=noise, size=adjusted_num_samples)
            
            # Stack the circular coordinates as the first two features
            circle_data = np.vstack((x_circle, y_circle)).T
            
            # Generate random noise for the remaining dimensions, if any
            if num_features > 2:
                additional_features = np.random.normal(
                    scale=noise, size=(adjusted_num_samples, num_features - 2)
                )
                # Concatenate the circle data with the additional noise features
                class_data = np.hstack((circle_data, additional_features))
            else:
                class_data = circle_data
            
            X.append(class_data)
            y += [i] * adjusted_num_samples  # Class labels
        
        X = np.vstack(X)
        y = np.array(y)
        
        return X, y
    

class MixedAndUnbalancedDataGenerator():
    def __init__(self):
        pass

    def generate(self,
                 num_samples: int, 
                 num_classes: int,
                 num_features:int,
                 random_state: int):
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Generate unbalanced probabilities for each class
        class_weights = np.random.dirichlet(np.ones(num_classes) * 1.1)
        
        # Randomly distribute samples among classes to create a more unbalanced dataset
        samples_per_class = np.random.multinomial(num_samples, class_weights)
        
        # Sort class indices based on the number of samples per class (ascending)
        sorted_indices = np.argsort(samples_per_class)
        
        # Define standard deviations (noise levels) for each class
        min_std, max_std = 0.5, 1.5
        std_devs = np.linspace(min_std, max_std, num_classes)
        ordered_std_devs = np.zeros_like(std_devs)
        ordered_std_devs[sorted_indices] = std_devs

        # Determine centers for each class
        # Classes with more samples are further from the origin
        max_distance = 5  # Maximum distance from the origin for the largest class
        distances = np.linspace(0.5, max_distance, num_classes)  # Distances for each class
        distances = distances[::-1]  # Reverse to assign larger distances to classes with more samples

        # Generate random directions for the centers in an n-dimensional space
        directions = np.random.randn(num_classes, num_features)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)  # Normalize to unit vectors

        # Scale directions by distances to determine centers
        centers = distances[:, np.newaxis] * directions

        # Reorder centers so that classes with more samples have larger distances from the origin
        ordered_centers = np.zeros_like(centers)
        ordered_centers[sorted_indices] = centers

        # Generate blobs with specified centers and standard deviations
        X, y = make_blobs(
            n_samples=samples_per_class,
            centers=ordered_centers,
            cluster_std=ordered_std_devs,
            n_features=num_features,
            random_state=random_state
        )
        
        return X, y