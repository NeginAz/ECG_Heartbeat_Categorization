import numpy as np

class DataAugmentation:
    def __init__(self, shift_max=5, noise_level=0.01, scale_range=(0.9, 1.1), expected_length=187):
        self.shift_max = shift_max
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.expected_length = expected_length
        
        # Define available augmentation methods (excluding random_crop)
        self.augmentations = [
            self.time_shift,
            self.add_noise,
            self.scale_signal
        ]

    def time_shift(self, signal):
        """Shift the signal in time by a small random value."""
        if self.shift_max > 0:
            shift = np.random.randint(-self.shift_max, self.shift_max)
            return np.roll(signal, shift)
        return signal

    def add_noise(self, signal):
        """Add subtle Gaussian noise to the signal."""
        noise = np.random.normal(0, self.noise_level, size=signal.shape)
        return signal + noise

    def scale_signal(self, signal):
        """Scale the signal amplitude slightly within the specified range."""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return signal * scale

    def random_augmentation(self, signal):
        """Apply a random augmentation method to the signal."""
        augmentation = np.random.choice(self.augmentations)
        augmented_signal = augmentation(signal)
        
        # Ensure the augmented signal has the correct shape
        if augmented_signal.shape[0] != self.expected_length:
            print(f"Incorrect shape: {augmented_signal.shape}. Expected: {self.expected_length}. Skipping this sample.")
            return signal  # Return the original signal if the shape is incorrect
        
        return augmented_signal

    def augment_batch(self, signals, augmentation_factor=0.5):
        """
        Augment a batch of signals.
        
        Parameters:
        - signals: array-like, the original signals to augment.
        - augmentation_factor: float, fraction of samples to augment (e.g., 0.5 for 50%).
        
        Returns:
        - np.array: Augmented signals.
        """
        augmented_signals = []
        
        # Calculate the number of samples to augment as an integer
        num_to_augment = int(len(signals) * augmentation_factor)
        
        for signal in signals[:num_to_augment]:
            # Generate a single augmented sample for each original signal
            augmented_signal = self.random_augmentation(signal)
            
            # Check if the augmented signal has the correct shape
            if augmented_signal.shape == (self.expected_length,):
                augmented_signals.append(augmented_signal)
            
        # Add the original unaugmented signals to the batch
        augmented_signals.extend(signals)
        
        return np.array(augmented_signals)









# import numpy as np

# class DataAugmentation:
#     def __init__(self, shift_max=5, noise_level=0.01, scale_range=(0.9, 1.1), crop_size=150, expected_length=187):
#         self.shift_max = shift_max
#         self.noise_level = noise_level
#         self.scale_range = scale_range
#         #self.crop_size = crop_size
#         self.expected_length = expected_length
        
#         # Define available augmentation methods
#         self.augmentations = [
#             self.time_shift,
#             self.add_noise,
#             self.scale_signal
#             #self.random_crop
#         ]

#     def time_shift(self, signal):
#         """Shift the signal in time by a small random value."""
#         if self.shift_max > 0:
#             shift = np.random.randint(-self.shift_max, self.shift_max)
#             return np.roll(signal, shift)
#         return signal
#     def add_noise(self, signal):
#         """Add subtle Gaussian noise to the signal."""
#         noise = np.random.normal(0, self.noise_level, size=signal.shape)
#         return signal + noise

#     def scale_signal(self, signal):
#         """Scale the signal amplitude slightly within the specified range."""
#         scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
#         return signal * scale

#     def random_crop(self, signal):
#         """Randomly crop a segment of the signal and pad the rest with zeros."""
#         if len(signal) <= self.crop_size:
#             return np.pad(signal, (0, self.expected_length - len(signal)), 'constant')
        
#         start = np.random.randint(0, len(signal) - self.crop_size)
#         cropped = signal[start:start + self.crop_size]
#         # Pad with zeros to maintain the expected length
#         return np.pad(cropped, (0, self.expected_length - len(cropped)), 'constant')



#     def random_augmentation(self, signal):
#         """Apply a random augmentation method to the signal."""
#         augmentation = np.random.choice(self.augmentations)
#         augmented_signal = augmentation(signal)
        
#         # Ensure the augmented signal has the correct shape
#         if augmented_signal.shape[0] != self.expected_length:
#             print(f"Incorrect shape: {augmented_signal.shape}. Expected: {self.expected_length}. Skipping this sample.")
#             return signal  # Return the original signal if the shape is incorrect
        
#         return augmented_signal

#     def augment_batch(self, signals, augmentation_factor=0.5):
#         """
#         Augment a batch of signals.
        
#         Parameters:
#         - signals: array-like, the original signals to augment.
#         - augmentation_factor: float, fraction of samples to augment (e.g., 0.5 for 50%).
        
#         Returns:
#         - np.array: Augmented signals.
#         """
#         augmented_signals = []
        
#         # Calculate the number of samples to augment as an integer
#         num_to_augment = int(len(signals) * augmentation_factor)
        
#         for signal in signals[:num_to_augment]:
#             # Generate a single augmented sample for each original signal
#             augmented_signal = self.random_augmentation(signal)
            
#             # Check if the augmented signal has the correct shape
#             if augmented_signal.shape == (self.expected_length,):
#                 augmented_signals.append(augmented_signal)
#             # else:
#             #     print(f"Skipping augmented signal with shape {augmented_signal.shape}")
            
#         # Add the original unaugmented signals to the batch
#         augmented_signals.extend(signals)
        
#         return np.array(augmented_signals)

