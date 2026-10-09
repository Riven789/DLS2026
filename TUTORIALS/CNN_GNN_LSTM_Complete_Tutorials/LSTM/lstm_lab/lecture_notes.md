# LSTM tutorial notes

## 1. Long short-term memory networks

An LSTM is a recurrent neural network that processes an ordered sequence while maintaining a hidden state and a cell state. It reuses the same weights at each time step. Gates decide how much of the previous cell state to retain, how much new information to write and how much of the cell state to expose as a hidden representation. This makes an LSTM useful when the order of observations affects the target.

The laboratory implements a modern LSTM with a forget gate directly in NumPy, with explicit backpropagation through time. It uses one recurrent layer, sixteen hidden units and one output for the whole sequence. A synthetic task asks for the last marked value, or its sign, among a sequence of numerical values and Boolean markers. This is a controlled memory exercise, not a realistic forecasting benchmark. The architecture has no peepholes, projection, bidirectionality or attention.

References
Hochreiter and Schmidhuber (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780. https://www.bioinf.jku.at/publications/older/2604.pdf
Gers, Schmidhuber and Cummins (2000). Learning to Forget: Continual Prediction with LSTM. https://pubmed.ncbi.nlm.nih.gov/11032042/
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html

## 2. A sequence has a time axis

A sequence tensor in this lab has shape [B,T,D]: batch, padded time length and input features. The model consumes one [B,D] slice at a time. The two input features are a real measurement and a Boolean marker. A validity mask distinguishes actual time steps from padding. The target is the value at the last marked time, or whether that value is positive.

In the illustrated sequence, two times carry markers, but only the later marked value determines the answer. A simple average of all measurements loses this ordering information. The final observation need not be marked. The network therefore has to learn when to update its memory and when to retain it. Dates, elapsed times and sampling intervals are not automatically understood by an LSTM. For irregular observations, add a meaningful elapsed-time feature or use an architecture that explicitly handles irregular time.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 3. Recurrence reuses one learned rule

Unrolling a recurrent model draws one copy of its computation for every time step. These copies share parameter arrays. Increasing the sequence length increases computation and activation storage, but it does not create a new set of weights for every observation. The hidden state represents information exposed to the next computation and output head. The cell state is a second recurrent pathway used to retain and update information.

The included model begins each independent sequence with zero hidden and cell states. It predicts once after the final valid time step. Training can shuffle independent sequences into minibatches while preserving order within each sequence. If a real application streams consecutive chunks of one long sequence, state may need to continue across chunks. Carrying state across unrelated users or examples would mix their information. Stateful streaming requires explicit state ownership and reset boundaries beyond this lab’s independent-sequence harness.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 4. Cell state and hidden state

A plain recurrent network repeatedly transforms its hidden state through a matrix and a nonlinearity. Long products of derivatives can shrink or grow, making long-range learning difficult. An LSTM adds a cell state whose update includes an additive pathway. This structure can provide a less attenuated gradient path when forget gates stay near one. It improves the mechanism for learning dependencies but does not guarantee unlimited or perfectly reliable memory.

Both h and c contain H values per example. They are learned representations with distributed meaning, not necessarily a list of human-readable facts. The cell state is not passed through tanh before it is stored, so it is not restricted to the interval from −1 to 1. In this basic variant, the hidden state equals an output gate times tanh(c), and therefore lies within that interval. The final task head applies another learned linear map to h.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 5. The four gate computations

At time t, concatenate the input x_t and previous hidden state h_(t−1) into q_t. One matrix multiplication produces four blocks of preactivations. Apply sigmoid to the input, forget and output blocks, and tanh to the candidate block. A single combined matrix is computationally convenient and equivalent to storing four separate matrices. The code’s block order is input, forget, candidate, output.

Each gate has H outputs. These are continuous, differentiable values rather than hard Boolean switches. A gate value near zero suppresses a contribution, and a value near one permits it. The numerical gate values depend on both current input and previous hidden state. The lab initializes the forget-gate bias to one, giving an initial tendency to retain memory when other preactivations are near zero. This is an initialization choice, not a rule that fixes the gate during training.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 6. The cell update combines retention and writing

The cell update has two contributions: f_t multiplied by the previous cell state, and i_t multiplied by the candidate g_t. Their sum forms the new cell state. The hidden state multiplies tanh of that cell state by the output gate. Gates are vectors, so different memory coordinates can retain, overwrite and expose different information at the same time.

The diagram focuses on the cell-state pathway. The input and previous hidden state determine the gates shown underneath, as established on the previous slide. The forget gate is not a reset button for the entire network. It can retain some coordinates while reducing others. Likewise, the output gate controls visibility rather than erasing the stored cell state. These equations describe the modern variant implemented in this tutorial. Earlier LSTM formulations and variants with peephole connections or projections use related but different equations.

References
Gers, Schmidhuber and Cummins (2000). Learning to Forget: Continual Prediction with LSTM. https://pubmed.ncbi.nlm.nih.gov/11032042/
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html

## 7. One memory coordinate calculated by hand

Suppose the previous cell coordinate equals 0.8. Let the forget gate be 0.9, the input gate 0.2 and the candidate −0.5. Retention contributes 0.72 and writing contributes −0.1, so the new cell coordinate is 0.62. If the output gate equals 0.7, the hidden coordinate is 0.7 times tanh(0.62), approximately 0.386.

The candidate can be negative, so writing can decrease as well as increase memory. Retaining most of the old state does not prevent a new contribution. If the input gate were zero and the forget gate one, this coordinate would preserve its old cell value. If the output gate were zero, the hidden coordinate would become zero while the cell could still retain its memory. Real gates come from learned weights applied to inputs and hidden state; these hand-picked values are an arithmetic illustration, not learned parameters from the reference experiment.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 8. Layers, directions, and parameter counts

For one LSTM layer with D input features and H hidden units, a combined input-and-hidden weight matrix has (D+H)×4H entries and the bias has 4H entries. The total is 4H(D+H+1). With D=2 and H=16 this equals 1,216. A two-class linear head adds 16×2+2=34 parameters, giving 1,250 in total. Scalar regression adds 17, giving 1,233.

A stacked LSTM sends each time step’s hidden representation from one recurrent layer to the next. The next layer generally receives H input features, so it needs a new recurrent parameter set. A bidirectional LSTM processes a complete sequence both forward and backward, often concatenating the two representations. It can use future context within the supplied window. That is useful for completed-sequence analysis but incompatible with strict online causality if future observations would be unavailable. The included implementation has one forward layer only.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 9. Labels determine the output pattern

A many-to-one model creates a single prediction for an entire input sequence. This is the pattern implemented in the lab. Its label array has one value for each input sequence. A many-to-many tagging task creates predictions aligned to input time steps and needs a loss mask so padded or unlabeled positions do not count. A sequence-to-sequence task may have a different output length and requires a decoder or another mapping beyond the simple final-state head here.

For forecasting, define the prediction time and horizon before creating windows. Inputs must contain only information available at that prediction time, and labels must refer to the intended future event. A sequence classifier can also be used on a fully observed interval. That is different from predicting the same interval’s label while only part of it is available. Supervised labels define the learning problem; the LSTM itself does not decide whether the task is causal, descriptive or predictive.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 10. Numerical, logical, and categorical inputs

The lab accepts two features per valid time step. Numerical values are standardized using only valid training time steps. The Boolean marker stays zero or one, preserving its logical meaning. A padding mask is a separate array; zero is also a legitimate numerical value and must not be used as an implicit padding indicator. Inputs and targets must be finite, and the validation function rejects incompatible shapes or marker values.

For other applications, encode categorical values with one-hot vectors or learned embeddings. Fit any category vocabulary on training data and define an unknown-category behavior. Impute missing numerical values using training-derived rules and include a missingness flag when its meaning matters. Elapsed time can be an additional numerical feature. These extensions change the schema and potentially the input dimension, so the supplied two-feature data adapter must be updated deliberately. Do not assume that padding, missing data and an observed zero mean the same thing.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 11. Variable length, padding, and minibatches

A minibatch can contain sequences with different lengths by padding each to a common T. The mask equals one at valid positions and zero after the end. At a valid time step the model applies the ordinary LSTM update. At a padding step it keeps both the previous cell state and previous hidden state unchanged. The final state therefore represents the last valid observation even if padding follows it.

Masking the input alone is insufficient because nonzero biases and the recurrent state can still change the next state when x is zero. The backward pass must also propagate the identity path through padding, while preventing a padded step from contributing parameter gradients. The lab uses right padding and rejects masks with a valid step after padding begins. A structural test appends extra masked steps containing arbitrary numbers and verifies that both predictions and parameter gradients remain unchanged. Grouping similar lengths reduces wasted computation in larger systems.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 12. Time splits and window leakage

The generated examples are independent synthetic sequences, so the lab uses a random 70/15/15 train/validation/test split, stratified for classification. This does not make random splitting appropriate for a real time series. Overlapping windows from one stream can share most of their observations, and future values can leak into preprocessing or target construction.

For forecasting, assign windows chronologically or by a justified rolling-origin protocol. Ensure training target windows finish before the validation period when required, and leave a gap large enough for overlapping inputs or labels to stop crossing the boundary. For multiple people, devices or sessions, use group boundaries when evaluating generalization to new entities. Fit normalization on training observations only. The CLI accepts an explicit per-example split array, allowing a caller to supply these assignments. The tutorial’s fixed random seed aids reproducibility but does not itself prevent information leakage.

References
scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 13. Classification and regression heads

The classification head produces two unrestricted logits, one for nonpositive and one for positive. Stable softmax converts logits to probabilities for reporting, and cross-entropy trains the model using integer class labels. A one-logit sigmoid head with binary cross-entropy would also be valid, but this lab consistently uses the two-logit formulation. Accuracy, macro-F1, per-class recall and a confusion matrix describe classification behavior.

The regression head produces one unrestricted real number and trains with mean squared error. Training target mean and scale standardize the regression objective, and predictions are converted back to original units before MAE, RMSE and R-squared are reported. A final sigmoid would restrict the output to zero through one and would be inappropriate for this signed target without a deliberate transformation. The same recurrent body can support both tasks, but each task is trained separately. The threshold defining the synthetic classification label is greater than zero.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 14. The training loop, learning rate, and AdamW

Each epoch shuffles complete sequences, forms batches of 32, computes a forward pass and loss, backpropagates through all time steps and updates parameters with AdamW. The learning rate scales optimizer steps. Too large a rate can cause unstable loss or repeated clipping; too small a rate may make useful learning slow. AdamW tracks first and second gradient moments with bias correction and applies decoupled weight decay. The supplied implementation excludes bias vectors from decay.

The reference run uses learning rate 0.003, decay 0.0001 and a global gradient-norm cap of 5. Clipping rescales all gradient tensors together when their combined norm exceeds the cap. It limits exploding gradients but does not repair incorrect derivatives. Validation loss chooses the saved weights and a patience of 15 can stop training before the 80-epoch cap. Test examples do not select the checkpoint. Checkpoints contain weights and inference metadata, not optimizer moments for exact training continuation.

References
Loshchilov and Hutter (2019). Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 15. Backpropagation through time

Backpropagation through time applies the chain rule to the unrolled recurrent computation. At each time, the gradient arriving through h contributes to the output gate and cell state, while the cell-state gradient also includes the contribution from later times. The direct cell-to-cell path multiplies by the forget gate. This is one derivative pathway, not the entire derivative, because gates also depend on earlier hidden states.

The derivatives of sigmoid and tanh are a(1−a) and 1−a² when written using their activation output a. Parameter gradients add across all time steps because the same W and b participate in each computation. The model accumulates these derivatives and performs one optimizer step per sequence batch. It uses full backpropagation through sequences of at most twelve time steps. Truncated backpropagation would cut the gradient across chunk boundaries, often while carrying numerical state forward. It changes which dependencies receive training signal and is an extension rather than a hidden feature of this code.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 16. An LSTM step in executable NumPy

The code excerpt uses one combined matrix to calculate gate preactivations. The gate order must match initialization, forward computation and backward computation. A common implementation error is to split the blocks in a different order in one of those places. Another is to overwrite the previous cell state before recording the values needed by the backward pass.

The actual model.py caches each step’s concatenated input, previous cell state, gate outputs, raw new cell and validity mask. Its reverse loop consumes these values and adds the shared weight gradients. The mask update shown in the complete implementation preserves h and c at padded positions. The excerpt includes mask updates. Here mask has shape [B,1] for the current time step. Seven tests cover sampled finite differences for every parameter tensor under both losses, split reproducibility, persistence, a hand-calculated cell and mask invariance. These checks establish the implemented arithmetic, while held-out evaluation tests whether the task was learned.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 17. Training, testing, and saved-model prediction

Run python run_all.py from the unzipped lab directory to execute tests, generate deterministic datasets, train both tasks, evaluate held-out data and write predictions for unlabeled examples. The shorter commands on this slide show how to run a second classification experiment in a new output directory. Use --task regression with the regression archive to train the scalar head. The README documents every command and input array.

Training stores weights.npz, metadata.json and history.json. Evaluation verifies the dataset checksum against the original labeled archive, restores the validation-selected checkpoint and writes test_metrics.json plus test_predictions.npz. Prediction accepts a compatible unlabeled archive and uses the saved normalization. Existing nonempty output directories are protected from accidental overwriting. Included reference_runs contain the completed experiments. A fresh run writes into runs, so reference evidence remains available for comparison. Seeded NumPy CPU results can vary slightly with numerical libraries or platform.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 18. Measured results on held-out sequences

The plots show actual classification cross-entropy and standardized regression mean squared error across epochs. Training curves use the full training split after each epoch; validation curves use only the validation split. The checkpoint with lowest validation loss is restored before test evaluation, so the classification result may correspond to an earlier epoch than the last point drawn.

The held-out classification set contains 242 sequences. Accuracy and macro-F1 are approximately 0.9835. The confusion matrix has 119 correct cases and 2 errors in each class. The regression test set contains 240 sequences and has RMSE approximately 0.02106 in the original signed-value units, compared with 0.57517 for predicting the training mean. These are seeded results on a small synthetic task. A direct algorithm that scans markers and returns the last marked value solves the task exactly. The exercise demonstrates learnable memory and a functioning harness, not a claim that an LSTM is needed for this rule.

References


## 19. Diagnostics and meaningful tests

Start by checking whether a small set of examples can be overfit. If not, inspect target alignment, state resets, tensor axes and gradients. Large or nonfinite losses can indicate a learning rate problem, an implementation bug or poorly scaled inputs. The logged maximum gradient norm is measured before clipping and can reveal whether the cap is frequently active, but it does not by itself identify the cause.

Test invariance to extra right padding, and test the opposite property for actual order: reordering valid observations can legitimately change the target. Compare performance as sequence lengths grow beyond training lengths rather than assuming learned memory extrapolates. A last-observation baseline is useful for some forecasting tasks, while the exact marker-scan rule is the relevant task-specific comparator here. When transferring the harness to real sequences, review overlap, entity identity and future information before interpreting a high test score.

References
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html

## 20. Laboratory and further reading

Begin by running the tests and inspecting one generated example with its mask and label. Follow the model through a valid time step, then through a padded one. Derive the parameter count and compare it with the shapes in the saved weights. Review the learning curves and use validation examples when comparing learning rates or hidden sizes.

Useful extensions include a simple recurrent-network baseline, different dependency lengths, a sequence-to-sequence loss with a label mask, or a second recurrent layer. Each changes the computation or target contract and deserves a corresponding structural or gradient check. For large datasets and long sequences, a mature automatic-differentiation library provides optimized kernels and better device support. Read the original LSTM and forget-gate papers for the historical motivations, and the modern LSTM equations for the exact variant discussed here. The companion lecture_notes.md contains the complete narration and source links for all twenty slides.

References
Hochreiter and Schmidhuber (1997). Long Short-Term Memory. Neural Computation 9(8), 1735-1780. https://www.bioinf.jku.at/publications/older/2604.pdf
Gers, Schmidhuber and Cummins (2000). Learning to Forget: Continual Prediction with LSTM. https://pubmed.ncbi.nlm.nih.gov/11032042/
PyTorch. LSTM documentation, modern input/forget/candidate/output equations. https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html
Goodfellow, Bengio and Courville (2016). Deep Learning, Chapter 10. https://www.deeplearningbook.org/contents/rnn.html