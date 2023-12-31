# -*- coding: utf-8 -*-
"""Layers that augment the functionality of a base layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from ..engine import Layer
from ..engine import InputSpec
from ..engine.topology import _object_list_uid
from ..utils.generic_utils import has_arg
from .. import backend as K


class Wrapper(Layer):
    """Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    # Arguments
        layer: The layer to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        self.layer = layer
        # Tracks mapping of Wrapper inputs to inner layer inputs. Useful when
        # the inner layer has update ops that depend on its inputs (as opposed
        # to the inputs to the Wrapper layer).
        self._input_map = {}
        super(Wrapper, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.built = True

    @property
    def activity_regularizer(self):
        if hasattr(self.layer, "activity_regularizer"):
            return self.layer.activity_regularizer
        else:
            return None

    @property
    def trainable(self):
        return self.layer.trainable

    @trainable.setter
    def trainable(self, value):
        self.layer.trainable = value

    @property
    def trainable_weights(self):
        return self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights

    @property
    def updates(self):
        if hasattr(self.layer, "updates"):
            return self.layer.updates
        return []

    def get_updates_for(self, inputs=None):
        # If the wrapper modifies the inputs, use the modified inputs to
        # get the updates from the inner layer.
        inner_inputs = inputs
        if inputs is not None:
            uid = _object_list_uid(inputs)
            if uid in self._input_map:
                inner_inputs = self._input_map[uid]

        updates = self.layer.get_updates_for(inner_inputs)
        updates += super(Wrapper, self).get_updates_for(inputs)
        return updates

    @property
    def losses(self):
        if hasattr(self.layer, "losses"):
            return self.layer.losses
        return []

    def get_losses_for(self, inputs=None):
        if inputs is None:
            losses = self.layer.get_losses_for(None)
            return losses + super(Wrapper, self).get_losses_for(None)
        return super(Wrapper, self).get_losses_for(inputs)

    def get_weights(self):
        return self.layer.get_weights()

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def get_config(self):
        config = {
            "layer": {
                "class_name": self.layer.__class__.__name__,
                "config": self.layer.get_config(),
            }
        }
        base_config = super(Wrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from . import deserialize as deserialize_layer

        layer = deserialize_layer(config.pop("layer"), custom_objects=custom_objects)
        return cls(layer, **config)


class TimeDistributed(Wrapper):
    """This wrapper applies a layer to every temporal slice of an input.

    The input should be at least 3D, and the dimension of index one
    will be considered to be the temporal dimension.

    Consider a batch of 32 samples,
    where each sample is a sequence of 10 vectors of 16 dimensions.
    The batch input shape of the layer is then `(32, 10, 16)`,
    and the `input_shape`, not including the samples dimension, is `(10, 16)`.

    You can then use `TimeDistributed` to apply a `Dense` layer
    to each of the 10 timesteps, independently:

    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
        # now model.output_shape == (None, 10, 8)
    ```

    The output will then have shape `(32, 10, 8)`.

    In subsequent layers, there is no need for the `input_shape`:

    ```python
        model.add(TimeDistributed(Dense(32)))
        # now model.output_shape == (None, 10, 32)
    ```

    The output will then have shape `(32, 10, 32)`.

    `TimeDistributed` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:

    ```python
        model = Sequential()
        model.add(TimeDistributed(Conv2D(64, (3, 3)),
                                  input_shape=(10, 299, 299, 3)))
    ```

    # Arguments
        layer: a layer instance.
    """

    def __init__(self, layer, **kwargs):
        super(TimeDistributed, self).__init__(layer, **kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = InputSpec(shape=input_shape)
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributed, self).build()

    def compute_output_shape(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        if has_arg(self.layer.call, "training"):
            kwargs["training"] = training
        uses_learning_phase = False

        input_shape = K.int_shape(inputs)
        if input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(x, _):
                global uses_learning_phase
                output = self.layer.call(x, **kwargs)
                if hasattr(output, "_uses_learning_phase"):
                    uses_learning_phase = (
                        output._uses_learning_phase or uses_learning_phase
                    )
                return output, []

            _, outputs, _ = K.rnn(
                step,
                inputs,
                initial_states=[],
                input_length=input_shape[1],
                unroll=False,
            )
            y = outputs
        else:
            # No batch size specified, therefore the layer will be able
            # to process batches of any size.
            # We can go with reshape-based implementation for performance.
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(inputs)[1]
            # Shape: (num_samples * timesteps, ...). And track the
            # transformation in self._input_map.
            input_uid = _object_list_uid(inputs)
            inputs = K.reshape(inputs, (-1,) + input_shape[2:])
            self._input_map[input_uid] = inputs
            # (num_samples * timesteps, ...)
            y = self.layer.call(inputs, **kwargs)
            if hasattr(y, "_uses_learning_phase"):
                uses_learning_phase = y._uses_learning_phase
            # Shape: (num_samples, timesteps, ...)
            output_shape = self.compute_output_shape(input_shape)
            y = K.reshape(y, (-1, input_length) + output_shape[2:])

        # Apply activity regularizer if any:
        if (
            hasattr(self.layer, "activity_regularizer")
            and self.layer.activity_regularizer is not None
        ):
            regularization_loss = self.layer.activity_regularizer(y)
            self.add_loss(regularization_loss, inputs)

        if uses_learning_phase:
            y._uses_learning_phase = True
        return y


class Bidirectional(Wrapper):
    """Bidirectional wrapper for RNNs.

    # Arguments
        layer: `Recurrent` instance.
        merge_mode: Mode by which outputs of the
            forward and backward RNNs will be combined.
            One of {'sum', 'mul', 'concat', 'ave', None}.
            If None, the outputs will not be combined,
            they will be returned as a list.

    # Raises
        ValueError: In case of invalid `merge_mode` argument.

    # Examples

    ```python
        model = Sequential()
        model.add(Bidirectional(LSTM(10, return_sequences=True),
                                input_shape=(5, 10)))
        model.add(Bidirectional(LSTM(10)))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```
    """

    def __init__(self, layer, merge_mode="concat", weights=None, **kwargs):
        if merge_mode not in ["sum", "mul", "ave", "concat", None]:
            raise ValueError(
                "Invalid merge mode. "
                "Merge mode should be one of "
                '{"sum", "mul", "ave", "concat", None}'
            )
        self.forward_layer = copy.copy(layer)
        config = layer.get_config()
        config["go_backwards"] = not config["go_backwards"]
        self.backward_layer = layer.__class__.from_config(config)
        self.forward_layer.name = "forward_" + self.forward_layer.name
        self.backward_layer.name = "backward_" + self.backward_layer.name
        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward_layer.initial_weights = weights[: nw // 2]
            self.backward_layer.initial_weights = weights[nw // 2 :]
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.return_state = layer.return_state
        self.supports_masking = True
        self._trainable = True
        super(Bidirectional, self).__init__(layer, **kwargs)
        self.input_spec = layer.input_spec

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        self.forward_layer.trainable = value
        self.backward_layer.trainable = value

    def get_weights(self):
        return self.forward_layer.get_weights() + self.backward_layer.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward_layer.set_weights(weights[: nw // 2])
        self.backward_layer.set_weights(weights[nw // 2 :])

    def compute_output_shape(self, input_shape):
        output_shape = self.forward_layer.compute_output_shape(input_shape)
        if self.return_state:
            state_shape = output_shape[1:]
            output_shape = output_shape[0]

        if self.merge_mode == "concat":
            output_shape = list(output_shape)
            output_shape[-1] *= 2
            output_shape = tuple(output_shape)
        elif self.merge_mode is None:
            output_shape = [output_shape, copy.copy(output_shape)]

        if self.return_state:
            if self.merge_mode is None:
                return output_shape + state_shape + copy.copy(state_shape)
            return [output_shape] + state_shape + copy.copy(state_shape)
        return output_shape

    def __call__(self, inputs, initial_state=None, **kwargs):
        if isinstance(inputs, list):
            if len(inputs) > 1:
                initial_state = inputs[1:]
            inputs = inputs[0]

        if initial_state is None:
            return super(Bidirectional, self).__call__(inputs, **kwargs)

        # Standardize `initial_state` into list
        if isinstance(initial_state, tuple):
            initial_state = list(initial_state)
        elif not isinstance(initial_state, list):
            initial_state = [initial_state]

        # Check if `initial_state` can be splitted into half
        num_states = len(initial_state)
        if num_states % 2 > 0:
            raise ValueError(
                "When passing `initial_state` to a Bidirectional RNN, the state "
                "should be a list containing the states of the underlying RNNs. "
                "Found: " + str(initial_state)
            )

        # Applies the same workaround as in `RNN.__call__`, without handling constants
        kwargs["initial_state"] = initial_state
        additional_inputs = initial_state
        additional_specs = [
            InputSpec(shape=K.int_shape(state)) for state in initial_state
        ]
        self.forward_layer.state_spec = additional_specs[: num_states // 2]
        self.backward_layer.state_spec = additional_specs[num_states // 2 :]

        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError(
                    "The initial state of a Bidirectional"
                    " layer cannot be specified with a mix of"
                    " Keras tensors and non-Keras tensors"
                    ' (a "Keras tensor" is a tensor that was'
                    " returned by a Keras layer, or by `Input`)"
                )

        if is_keras_tensor:
            # Compute the full input spec, including state
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs

            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(Bidirectional, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(Bidirectional, self).__call__(inputs, **kwargs)

    def call(self, inputs, training=None, mask=None, initial_state=None):
        kwargs = {}
        if has_arg(self.layer.call, "training"):
            kwargs["training"] = training
        if has_arg(self.layer.call, "mask"):
            kwargs["mask"] = mask

        if initial_state is not None and has_arg(self.layer.call, "initial_state"):
            forward_state = initial_state[: len(initial_state) // 2]
            backward_state = initial_state[len(initial_state) // 2 :]
            y = self.forward_layer.call(inputs, initial_state=forward_state, **kwargs)
            y_rev = self.backward_layer.call(
                inputs, initial_state=backward_state, **kwargs
            )
        else:
            y = self.forward_layer.call(inputs, **kwargs)
            y_rev = self.backward_layer.call(inputs, **kwargs)

        if self.return_state:
            states = y[1:] + y_rev[1:]
            y = y[0]
            y_rev = y_rev[0]

        if self.return_sequences:
            y_rev = K.reverse(y_rev, 1)
        if self.merge_mode == "concat":
            output = K.concatenate([y, y_rev])
        elif self.merge_mode == "sum":
            output = y + y_rev
        elif self.merge_mode == "ave":
            output = (y + y_rev) / 2
        elif self.merge_mode == "mul":
            output = y * y_rev
        elif self.merge_mode is None:
            output = [y, y_rev]

        # Properly set learning phase
        if getattr(y, "_uses_learning_phase", False) or getattr(
            y_rev, "_uses_learning_phase", False
        ):
            if self.merge_mode is None:
                for out in output:
                    out._uses_learning_phase = True
            else:
                output._uses_learning_phase = True

        if self.return_state:
            if self.merge_mode is None:
                return output + states
            return [output] + states
        return output

    def reset_states(self):
        self.forward_layer.reset_states()
        self.backward_layer.reset_states()

    def build(self, input_shape):
        with K.name_scope(self.forward_layer.name):
            self.forward_layer.build(input_shape)
        with K.name_scope(self.backward_layer.name):
            self.backward_layer.build(input_shape)
        self.built = True

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            if not self.merge_mode:
                return [mask, mask]
            else:
                return mask
        else:
            return None

    @property
    def trainable_weights(self):
        if hasattr(self.forward_layer, "trainable_weights"):
            return (
                self.forward_layer.trainable_weights
                + self.backward_layer.trainable_weights
            )
        return []

    @property
    def non_trainable_weights(self):
        if hasattr(self.forward_layer, "non_trainable_weights"):
            return (
                self.forward_layer.non_trainable_weights
                + self.backward_layer.non_trainable_weights
            )
        return []

    @property
    def updates(self):
        if hasattr(self.forward_layer, "updates"):
            return self.forward_layer.updates + self.backward_layer.updates
        return []

    @property
    def losses(self):
        if hasattr(self.forward_layer, "losses"):
            return self.forward_layer.losses + self.backward_layer.losses
        return []

    @property
    def constraints(self):
        constraints = {}
        if hasattr(self.forward_layer, "constraints"):
            constraints.update(self.forward_layer.constraints)
            constraints.update(self.backward_layer.constraints)
        return constraints

    def get_config(self):
        config = {"merge_mode": self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
