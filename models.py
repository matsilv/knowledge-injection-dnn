# @Author: Mattia Silvestri

"""
    Implementation of SBR-inspired loss function in TensorFlow 2.
"""

import tensorflow as tf
import numpy as np

from utility import compute_feasibility_from_predictions, visualize


########################################################################################################################

class MyModel(tf.keras.Model):
    def __init__(self, num_layers, num_hidden, input_shape, output_dim, method='agnostic', lmbd=1.0):
        """
        Abstract class implementing fully-connected feedforward NN.
        :param num_layers: number of hidden layers; as integer
        :param num_hidden: number of hidden units for each layer; as a list of integers
        :param input_shape: input shape required by tf.keras; as a tuple
        :param output_dim: number of output neurons; as integer
        :param method: method to be applied to the NN; as string
        :param lmbd: lambda for SBR-inspired loss term
        """

        super(MyModel, self).__init__(name="mymodel")

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.output_dim = output_dim

        available_methods = ['agnostic', 'sbrinspiredloss', 'confidences']
        if method not in available_methods:
            raise Exception("Method selected not valid")
        self.method = method

        # Lambda for SBR-inspired loss term
        self.lmbd = lmbd

        # build the neural net model
        self.__define_model__(input_shape)

        # define the optimizer
        self.__define_optimizer__()

    def __define_model__(self, input_shape):

        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Dense(self.num_hidden[0], activation=tf.nn.relu, input_shape=input_shape))
        for i in range(1, self.num_layers):
            self.model.add(tf.keras.layers.Dense(self.num_hidden[i], activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dense(self.output_dim))

    def __define_optimizer__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    #@tf.function
    def grad(self, inputs, targets, penalties, weights):
        """
        Compute loss and gradients.
        :param inputs: input instances
        :param targets: target instances
        :param penalties: penalties instances
        :param weights: weights to solve balancing issues
        :return: loss values and gradients
        """

        with tf.GradientTape() as tape:
            loss_value, cross_entropy_loss, sbr_inspired_loss = \
                self.compute_loss(inputs, targets, penalties, weights)

        grads = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_value, cross_entropy_loss, sbr_inspired_loss

    def compute_loss(self, tensor_X, tensor_y, tensor_p, tensor_w):
        """
        Compute SBR loss function.
        :param tensor_X: input instances as tf.Tensor with shape=(batch_size, n**3)
        :param tensor_y: instances' labels as tf.Tensor of shape=(batch_size, n**3)
        :param tensor_p: penalties as tf.Tensor of shape=(batch_size, n**3)
        :param tensor_w: weights as tf.Tensor of shape=(batch_size, n**3)
        :return: loss value
        """

        # each element is 1 if that value in that position cannot be assigned, 0 otherwise
        tensor_p = tf.cast(tensor_p, dtype=tf.float32)
        tensor_y = tf.cast(tensor_y, dtype=tf.float32)
        tensor_X = tf.cast(tensor_X, dtype=tf.float32)
        tensor_w = tf.cast(tensor_w, dtype=tf.float32)

        y_pred = self.model(tensor_X)

        # supervised loss function
        cross_entropy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tensor_y, y_pred, from_logits=True))

        # SBR inspired loss
        sbr_inspired_loss = tf.reduce_mean(tf.reduce_sum(tf.square((1 - tensor_p) - tf.nn.sigmoid(y_pred)), axis=1))

        idx = 0
        p_numpy = tensor_p.numpy()
        y_numpy = tensor_y.numpy()
        x_numpy = tensor_X.numpy()
        w_numpy = tensor_w.numpy()

        p_numpy = p_numpy[idx].reshape(10, 10, 10)
        w_numpy = w_numpy[idx].reshape(10, 10, 10)
        print('--------------------- Partial solution ----------------------')
        visualize(x_numpy[idx].reshape(10, 10, 10))
        print()
        print('--------------------- Assignment ---------------------')
        visualize(y_numpy[idx].reshape(10, 10, 10))
        print()
        print('--------------------- Penalties ------------------------')
        for i in range(10):
            for j in range(10):
                print(p_numpy[i, j])
            print()

        print()
        print('--------------------- Weights ---------------------------')
        for i in range(10):
            for j in range(10):
                print(w_numpy[i, j])
            print()

        bce = tf.nn.sigmoid_cross_entropy_with_logits(tensor_p, y_pred) * tensor_w
        bce_numpy = bce.numpy()
        bce_numpy = bce_numpy[idx].reshape(10, 10, 10)
        print()
        print('------------------- Binary cross-entropy ---------------------')
        for i in range(10):
            print(bce_numpy[i])
            print()

        label = np.argmax(y_numpy)
        label = np.unravel_index(label, (10, 10, 10))
        print(label)
        print(bce_numpy[label])
        exit()

        # binary cross-entropy
        binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tensor_p, y_pred) * tensor_w)

        ''''binary_cross_entropy = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tensor_p, y_pred, from_logits=True)) '''

        if self.method == 'sbrinspiredloss':
            print(self.lmbd)
            loss = cross_entropy_loss + sbr_inspired_loss * self.lmbd
        elif self.method == 'agnostic':
            loss = cross_entropy_loss
        else:
            loss = binary_cross_entropy

        return loss, cross_entropy_loss, sbr_inspired_loss

    @tf.function
    def predict_from_saved_model(self, X):
        """
        Make predictions given input instances from tf 2 SavedModel.
        :param X: input instances as numpy array with shape=input_shape
        :return: numpy array of shape=(?, num_classes)
        """

        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)

        # Keras signatures is serving default
        infer = self.model.signatures["serving_default"]
        # make inference
        pred_tensor = infer(X_tensor)

        # inference output is a dictionary; last layer is the output one
        pred_tensor = pred_tensor["dense_{}".format(self.num_layers)]

        # output layer returns logits
        pred_tensor = tf.nn.softmax(tf.cast(pred_tensor, dtype=tf.float32))

        return pred_tensor

    def train(self, num_epochs, train_ds, ckpt_dir, dim, val_set, use_prop):
        """
        Train the model.
        :param num_epochs: number of training epochs
        :param train_ds: training set as tf.Dataset
        :param ckpt_dir: training checkpoint directory; as string
        :param dim: PLS dimension; as integer
        :param val_set: validation dataset; as tuple of 2 numpy array representing inputs and penalties
        :param use_prop: use propagation during validation
        :return: losses as dictionary of lists
        """

        # Keep track of losses and accuracy results
        loss_history = []
        cross_entropy_loss_history = []
        sbr_inspired_loss_history = []
        train_accuracy_results = []
        history = {}

        # Keep track of feasibility results for validation and number of not improved epochs
        best_feas = 0
        count_not_improved = 0

        # Variables for checkpointing
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        manager = tf.train.CheckpointManager(ckpt, '{}/tf_ckpts'.format(ckpt_dir), max_to_keep=3)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        # Training epochs
        for epoch in range(num_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_cross_entropy_loss_avg = tf.keras.metrics.Mean()
            epoch_sbr_inspired_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

            # Training loop - using batches
            for x, y, p, w in train_ds:

                '''idx = 20

                x_numpy = x.numpy()
                y_numpy = y.numpy()
                p_numpy = p.numpy()
                w_numpy = w.numpy()
                p_numpy = p_numpy[idx].reshape(10, 10, 10)
                w_numpy = w_numpy[idx].reshape(10, 10, 10)
                visualize(x_numpy[idx].reshape(10, 10, 10))
                print()
                visualize(y_numpy[idx].reshape(10, 10, 10))
                print()
                for i in range(10):
                    for j in range(10):
                        print(p_numpy[i,j])
                    print()

                print()
                for i in range(10):
                    for j in range(10):
                        print(w_numpy[i, j])
                    print()
                exit(0)'''

                loss_value, cross_entropy_loss, sbr_inspired_loss = self.grad(x, y, p, w)

                # Track progress
                epoch_loss_avg(loss_value)  # Add current batch loss
                epoch_cross_entropy_loss_avg(cross_entropy_loss)
                epoch_sbr_inspired_loss_avg(sbr_inspired_loss)
                # Compare predicted label to actual label
                epoch_accuracy(y, self.model(x).numpy())

            # End epoch
            loss_history.append(epoch_loss_avg.result().numpy())
            cross_entropy_loss_history.append(epoch_cross_entropy_loss_avg.result().numpy())
            sbr_inspired_loss_history.append(epoch_sbr_inspired_loss_avg.result().numpy())
            train_accuracy_results.append(epoch_accuracy.result().numpy())

            # Save checkpoint every 10 epochs and compute validation feasibility
            if (epoch + 1) % 10 == 0:

                if val_set is not None:
                    x_val = val_set[0]
                    p_val = val_set[1]

                    '''visualize(x_val[120].reshape(10, 10, 10))
                    p_val = p_val[120].reshape(10, 10, 10)
                    print()
                    for i in range(10):
                        for j in range(10):
                            print(p_val[i,j])
                        print()'''

                    preds = self.model(x_val)

                    if use_prop:
                       preds = preds * (1 - p_val)

                    feas = compute_feasibility_from_predictions(x_val, preds, dim)
                    print("Current feasibility: {} | Best feasibility: {}".format(feas, best_feas))

                    # If last checkpoint validation feasibility was higher than current one, then stop training
                    if feas <= best_feas:
                        count_not_improved += 1
                        print("{} times the feasibility has not improven".format(count_not_improved))
                    else:
                        best_feas = feas
                        count_not_improved = 0
                        # Save checkpoint
                        save_path = manager.save()
                        tf.saved_model.save(self.model, ckpt_dir)
                        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

                    if count_not_improved == 5:
                      break

            print(
                "Epoch {:03d}: Loss: {:.5f} | Cross entropy loss: {:.5f}, SBR inspired loss: {:.8f}, Accuracy: {:.5%}".format(
                    epoch,
                    epoch_loss_avg.result(),
                    epoch_cross_entropy_loss_avg.result(),epoch_sbr_inspired_loss_avg.result(),
                    epoch_accuracy.result()))

        # save a dictionary with epochs losses
        history["loss"] = loss_history
        history["cross_entropy_loss"] = cross_entropy_loss_history
        history["sbr_inspired_loss"] = sbr_inspired_loss_history

        return history

