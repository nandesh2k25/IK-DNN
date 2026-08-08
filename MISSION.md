# Mission: Minimizing Error in Deep Neural Networks (applied to IK_MDN_3R_3D)

## Why
I finished Course 1 of Andrew Ng's Deep Learning Specialization and now want to reduce the prediction error of my thesis model (`MDN3R3D`, a Mixture Density Network solving 3R robot inverse kinematics in 3D). I need the Course 2-level concepts — diagnosing bias/variance, regularization, optimization, hyperparameter tuning — taught against my own model's code and config, not abstract textbook examples.

## Success looks like
- I can look at my model's train/val loss curves and correctly diagnose whether I'm underfitting, overfitting, or well-fit.
- I can pick the correct config knob (`hidden_size`, `n_components`, `lambda_fk`, `learning_rate`, regularization) to address whatever regime I'm in, and explain why.
- I understand what my optimizer (Adam) and existing safeguards (gradient clipping) are actually doing, well enough to tune them deliberately instead of by guesswork.
- I can add missing Course 2 techniques (dropout, weight decay, learning-rate scheduling) to `MDN3R3D` correctly, when the diagnosis calls for it.

## Constraints
- Short, frequent lessons (10-15 min each) — fits between other work, not long study sessions.
- Grounded in PyTorch (the project's framework) — skip TensorFlow/Keras mechanics from Course 2.
- Prioritize whatever unblocks progress on the thesis model over completeness of the course curriculum.

## Out of scope (for now)
- TensorFlow/Keras framework mechanics from Course 2.
- Full mathematical derivations of optimizer convergence proofs (light intuition is enough).
- Course 3+ material (ML project structuring, CNNs, sequence models) unless the thesis model later needs it.
