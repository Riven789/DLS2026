# CNN tutorial notes

## 1. Convolutional neural networks

A convolutional neural network applies learned local computations repeatedly across a spatial grid. This tutorial continues the MLP treatment: weights, nonlinear activations, losses, gradients and optimizers remain the ingredients, while connectivity and parameter sharing change. We use images as the main example, but the same idea extends to one-dimensional signals and three-dimensional volumes.

The laboratory implements two convolution blocks directly in NumPy, including gradients, average pooling, global average pooling, training, held-out testing and saved-model prediction. The data are synthetic noisy bars with different orientations and amplitudes. One task predicts orientation and a second predicts amplitude. These controlled tasks demonstrate the implementation; they are not evidence of performance on real microscopy or detector data. All code runs on a CPU with no dataset downloads after NumPy is installed.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 2. Images are arrays with spatial meaning

An image is a grid of measurements. A grayscale image has one channel; an RGB image has three channels whose values refer to different color components at the same location. Scientific images may have spectral channels, physical variables, or time slices, with units and conventions that must be recorded. A batch adds an axis for independent images.

The lab uses NCHW order: examples, channels, height, width. Other libraries may use NHWC. A silent channel-order mistake can produce valid array operations with meaningless results. Pixel intensities become floating-point values before training. Fit channel means and scales using training pixels only, then reuse those quantities. Do not normalize each image independently if absolute brightness carries the target signal, as it does in the amplitude regression example. Pixel neighbors should represent actual spatial neighbors.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 3. Local connections and shared filters

A dense layer connects each output to every input position with independent weights. A convolution restricts each output to a local patch and reuses the same weights at every allowed position. This supplies an inductive bias: the same local pattern can matter wherever it occurs. The network learns a detector without learning a different detector for each location.

The feature map records the response at each location. A bias is normally shared across all spatial positions within an output channel. Each additional filter learns another pattern and produces another output channel. The drawing is a numerical grid rather than a biological analogy. Real learned filters need not resemble obvious edges, and deeper filters operate over feature maps, not directly over original pixels. Spatial weight sharing reduces parameter count, but the convolution still performs work at every output position.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 4. A convolution calculated by hand

Place the two-by-two kernel on the upper-left input patch. Multiply corresponding elements and sum: 1×1 + 2×0 + 0×0 + 1×(−1)=0. Move one column to the right and repeat to get −1. The bottom-left output is −1 and the bottom-right output is 1. There is no padding and the stride is one, so a three-by-three input produces a two-by-two output.

The operation shown is technically cross-correlation because the kernel is not reversed. Neural-network APIs conventionally call it convolution. Since the kernel values are learned, this naming convention does not prevent the model from learning the desired transformation. The code and finite-difference tests follow the unflipped convention consistently. A separate bias would be added to each output response; it is zero in this example. This exact calculation appears in the CNN tests.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 5. Stride, padding, and output size

For one spatial dimension, output length is floor((input + 2 padding − dilation×(kernel−1) −1)/stride)+1. The displayed equation assumes dilation one. Apply it independently to height and width. Symmetric zero padding adds a border of zeros. Padding can preserve resolution for odd kernels with stride one, but the artificial border also changes responses near image edges.

Stride two samples outputs every two input positions, reducing resolution. Pooling can reduce resolution as a separate operation. Dilation spreads the kernel taps farther apart without increasing the number of learned weights. These are architectural choices rather than learned parameters. The lab deliberately implements stride-one, dilation-one convolutions with a three-by-three kernel and padding one. Its only spatial downsampling is a two-by-two average pool. The lecture explains additional choices so learners can interpret larger CNNs without suggesting those choices are all implemented.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 6. Channels and trainable parameters

For a standard dense convolution, weights have shape output channels by input channels by kernel height by kernel width. To create one output channel, sum the local weighted responses over every input channel, then add the filter bias. Output channels have separate filters but each filter reuses its own weights across positions.

Six three-by-three filters on a one-channel image have 6×1×3×3=54 weights and 6 biases, or 60 parameters. The next layer takes six channels and produces eight, so it contains 8×6×3×3+8=440 parameters. These counts are independent of image height and width, unlike a fully connected layer on flattened pixels. Groups and depthwise convolutions change this connectivity and parameter formula; they are useful extensions but are not part of this lab. Batch size changes activation storage rather than the number of trainable parameters.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 7. Nonlinear feature maps

Convolution alone is linear in the input when bias is zero and affine when bias is included. Composing affine layers without a nonlinearity still gives an affine function. ReLU adds the ability to select different local patterns under different input conditions. It keeps positive responses and replaces negative responses with zero.

A feature map has both a channel identity and a spatial coordinate. The example matrix is an illustrative response pattern, not a trained filter from the lab. After ReLU, some locations become inactive. During backpropagation, the derivative is one for positive preactivations and zero for negative ones; this implementation chooses zero at exactly zero. Keep the original preactivation for calculating that mask. A final classification layer usually returns unconstrained logits rather than a ReLU output, since negative scores remain meaningful relative to other class scores.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 8. Pooling summarizes nearby responses

A two-by-two average pool with stride two replaces four numbers by their mean. On the example patch 1,3,2,6, average pooling returns 3 while max pooling returns 6. Pooling has no learned weights in this form. It reduces activation memory and the resolution seen by subsequent layers.

Average-pool backpropagation divides the incoming gradient equally among the four inputs. Max-pool backpropagation routes it to the selected maximum, with a defined tie convention if values are equal. The lab implements average pooling because its derivative is transparent. Pooling trades fine position information for coarser summaries. It does not create exact invariance to every translation, and excessive downsampling can remove a small feature entirely. For segmentation or precise localization, preserve or reconstruct spatial detail instead of assuming a pooled representation is sufficient.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 9. Receptive fields grow across layers

The receptive field of an activation is the set of input positions that can influence it. For a sequential stack, track the receptive-field width r and the jump j between adjacent activations measured in input coordinates. Initially r=1 and j=1. A layer with effective kernel k and stride s gives r_new=r+(k−1)j and j_new=js.

The lab begins with a three-by-three convolution, giving r=3. The two-by-two pool expands this to r=4 and increases the jump to 2. A second three-by-three convolution gives r=4+2×2=8. Before global pooling, each feature-map value therefore has a theoretical eight-by-eight receptive field, subject to border padding. Global average pooling combines locations across the complete map. The theoretical footprint does not imply equal sensitivity to every pixel within it. Learned weights and nonlinear activation masks determine actual influence.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 10. The complete laboratory CNN

The entire architecture can be understood by following shape changes. Start with B one-channel twelve-by-twelve images. The first convolution keeps spatial size and creates six channels. ReLU preserves that shape. Average pooling reduces each spatial dimension to six. The second convolution creates eight channels at six-by-six resolution, followed by ReLU. Global averaging reduces each channel to one number, so the final representation has shape B by eight.

The classification head maps eight features to three logits. It contributes 8×3+3=27 parameters. Added to 60 and 440 convolution parameters, this yields 527 total. The regression head maps eight features to one scalar and contains nine parameters, for a total of 509. Global averaging accepts other even image sizes supported by the pooling operation, but changes in resolution or field of view can still shift the data distribution. Shape compatibility is not a guarantee of generalization.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 11. Classification and regression heads

The classification dataset labels noisy images as horizontal, vertical or diagonal. Its final dense layer emits three logits, softmax converts those scores to class probabilities, and cross entropy measures the probability assigned to the correct class. Class IDs are 0,1,2. The training loss uses stable log-softmax arithmetic directly on logits, with no extra softmax passed into the loss.

The regression dataset asks for the bar amplitude, a continuous value. It uses one linear output and mean squared error. Training standardizes the target with training-only statistics, while evaluation converts predictions back to original amplitude units before calculating MAE, RMSE and R². The two datasets are separately generated examples and each has its own split and checkpoint. Probabilities are not automatically calibrated, and the regression head does not enforce a physical range. Those requirements would need deliberate output choices or calibration.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 12. Image data and labels

The lab accepts compressed NumPy archives instead of depending on an image-loading framework. X is a finite numeric array with shape N,C,H,W. y has one scalar per example: an integer class ID or a continuous regression target. An optional split array assigns each example to train, validation or test. Unlabeled prediction archives omit y.

Convert image files upstream with a documented resizing, channel-order and intensity convention. Different interpolation methods affect small features, and a contrast normalization step can destroy an amplitude target. Missing or corrupt pixels need a task-appropriate treatment, often with a separate validity channel; this minimal loader rejects nonfinite values. Boolean masks can become 0/1 channels. Scalar metadata such as instrument mode is often encoded separately and concatenated to the pooled representation, rather than painted into an artificial image. That metadata branch is an extension, not implemented here.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 13. Splits and augmentation

Random splitting is suitable for the independently generated images in this lab. In real data, split by patient, specimen, experiment, scene, or acquisition session when images from the same source are correlated. Crops or augmented copies of the same image should not appear in different splits. A test set that contains near duplicates can make performance look much better than it will be on new experiments.

Training-only augmentation can expose useful variations. Translations may preserve an orientation label, while a ninety-degree rotation changes horizontal into vertical. A brightness adjustment changes the amplitude regression target unless the target is transformed consistently. The lab varies location, orientation, amplitude and noise in its generator but does not include an additional online augmentation pipeline. Validation and test transforms should be deterministic and documented. Fit normalization on training pixels, use validation for model choices, and inspect the test metrics after those choices are fixed.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 14. Backpropagation through shared weights

Each kernel coefficient participates in many output calculations. The derivative with respect to that coefficient is the sum of the local input value times the loss derivative at the corresponding output. Summing over both images and spatial positions is essential. A bias receives the summed output derivative for its channel. The mean-over-batch factor already enters through the output loss derivative.

For the input gradient, each output distributes its derivative back to the input patch through the filter weights. Overlapping patches accumulate into the same input locations. The code performs that accumulation explicitly over kernel offsets and removes the padded border afterward. ReLU masks block gradients where preactivations are nonpositive. Average pooling distributes them equally. Central finite differences check sampled entries in every parameter tensor for both classification and regression. Another exact test checks overlap accumulation in a three-by-three convolution.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 15. The training loop and optimizer

The training harness shuffles training examples once per epoch, retains the incomplete final batch, and computes a fresh set of gradients for each update. AdamW uses learning rate 0.003 and weight decay 0.0001 by default. Biases are excluded from decay. A global gradient norm cap of 5 is available; clipping is a safeguard and does not repair a fundamentally excessive learning rate.

After each epoch, evaluate complete training and validation sets in batches with example-weighted loss accumulation. Save copies of parameters whenever validation improves. Early stopping waits fifteen epochs without a sufficient improvement, and the best weights are restored at the end. Test images never select an epoch or hyperparameter. The NumPy model has no dropout or batch-normalization state, so training and evaluation share the same mathematical forward function. Larger CNNs often include such layers and require explicit training versus evaluation modes.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/
Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101

## 16. Convolution in executable NumPy

The forward convolution first pads the input, then uses sliding_window_view to expose all local kernel-sized patches. This creates a strided view rather than manually copying each patch in a Python loop. The resulting patch axes are batch, input channel, output row, output column, kernel row and kernel column. einsum sums over channel and kernel coordinates to create output channels at every spatial position.

The excerpt uses symbolic W and b for one layer. model.py contains the full implementation and caches the information required for backward computation. It supports the exact stride-one convolutions used in this laboratory. This code is designed for transparency rather than competitive convolution performance. Specialized libraries use highly optimized CPU or accelerator kernels and automatic differentiation. Understanding the tensor contraction remains useful when inspecting such implementations or debugging shape errors.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 17. Training, testing, and saved-model prediction

run_all.py runs the seven tests, generates data, trains the classification and regression examples, evaluates their frozen test splits, and predicts six unlabeled examples. The independent commands in lab.py let learners change one hyperparameter at a time. Each experiment needs a new output directory so previous results are not overwritten.

weights.npz contains learned parameters. metadata.json records architecture, input normalization, target scaling, class names, split indices, source-data hash, settings and software versions. history.json stores learning curves, and test_metrics.json stores results from the separate evaluation command. The data hash prevents reuse of test indices on a modified archive. predict accepts a different unlabeled archive and reuses the saved transformations. These checkpoints support inference, not exact mid-run resumption: optimizer moments and random-generator state are not included.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 18. Measured results on held-out examples

These plots and values are read from the saved reference run, rather than invented as illustrative curves. The generator uses seed 42 and the model and split use seed 7. The data contain noisy twelve-by-twelve images with horizontal, vertical or diagonal bars. Locations and amplitudes vary. The regression target is the generated bar amplitude.

Classification uses a majority-class baseline derived only from training labels. Regression uses the training target mean. Those are minimal reference points rather than strong competing models. The examples are intentionally simple enough to inspect, and high accuracy does not imply comparable performance on real scientific images. The learning curve shows all completed epochs; the evaluated checkpoint is the one with the lowest validation loss. Full confusion matrices, test predictions, software versions and weights accompany the lab. Changing numerical libraries may produce small differences.

References


## 19. Diagnostics and useful tests

Check examples visually with their labels before training. A transposed or misordered tensor can still pass through the network while representing the wrong input. Test the exact convolution and pooling arithmetic on tiny arrays before relying on end-to-end learning. The package includes sampled finite-difference parameter checks for both losses, checkpoint roundtrips, split checks, training-only normalization, stable losses and exact convolution/pooling examples.

A CNN is translation equivariant under appropriate idealized conditions, but boundaries, stride and pooling complicate the statement. Global pooling promotes tolerance to location without guaranteeing exact invariance to arbitrary shifts. Evaluate shifts and resolutions that are relevant to the actual application. Overfitting may call for stronger regularization or better data coverage, while poor training performance may indicate scaling or gradient problems. Saliency maps and filter pictures can be suggestive, but they do not independently establish causal explanations.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/

## 20. Laboratory and further reading

Install NumPy in a virtual environment and run the one-command laboratory. Read model.py alongside slides 4, 8 and 14 to connect local operations to their gradients. Study common.py and lab.py to connect optimization, validation selection and held-out evaluation. README.md documents the data contract, custom-data procedure and known limitations.

Exercises can be completed incrementally. Replace average pooling with max pooling and implement its backward pass. Add a stride option to convolution and verify output shapes and gradients. Compare a flattened dense head with global averaging while keeping the split fixed. Test translations, rotations and intensity transformations, identifying which transformations preserve each target. Keep hyperparameter selection on validation data and reserve test data for a final comparison. Consult the primary teaching sources below for larger architectures and the mathematical treatment of convolution.

References
Stanford CS231n. Convolutional Neural Networks. https://cs231n.github.io/convolutional-networks/
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 9. https://www.deeplearningbook.org/contents/convnets.html
Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101
scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html