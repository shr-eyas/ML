'''
More stable and accurate results can be obtained if the softmax and loss are combined during training. 
This is enabled by the 'preferred' organization shown here.
In the preferred organization the final layer has a linear activation. 
For historical reasons, the outputs in this form are referred to as logits. 
The loss function has an additional argument: from_logits = True. 
This informs the loss function that the softmax operation should be included in the loss calculation. 
This allows for an optimized implementation.
'''
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs

centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #<-- Note
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10         #similar to no of iterations
)
    
