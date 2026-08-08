# Deep Learning Error-Minimization Resources

## Knowledge

- [Course: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization — Andrew Ng / DeepLearning.AI (Coursera)](https://www.coursera.org/learn/deep-neural-network/)
  Direct continuation of Course 1, same specialization and teaching style already trusted. Use for: bias/variance framework, regularization (L2, dropout), optimization algorithms (momentum, RMSprop, Adam), batch norm, hyperparameter search strategy. Primary source for this whole workspace.
- [CS231n Notes: Neural Networks Part 2 (Setting up the Data and the Loss) — Stanford](https://cs231n.github.io/neural-networks-2/)
  Stanford's freely available notes on regularization (L2, dropout, maxnorm) with sharp intuition. Use for: a second explanation of regularization if Course 2's framing doesn't land.
- [Distill.pub: "Why Momentum Really Works" — Gabriel Goh](https://distill.pub/2017/momentum/)
  Interactive, visual explanation of why momentum-based optimizers converge faster than vanilla gradient descent. Use for: real intuition for what Adam does under the hood, beyond "it just works."
- [PyTorch docs: torch.optim.Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
  Authoritative reference for the exact optimizer already used in `train_mdn` (lr, betas, eps, weight_decay). Use for: correctly setting `weight_decay` (L2 regularization) on the real model.
- [PyTorch docs: torch.nn.Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
  Authoritative reference for adding dropout to `MDN3R3D`'s backbone. Use for: implementing dropout correctly (train/eval mode behavior) once it's needed.

## Wisdom (Communities)

- Gap: no community identified yet. Revisit once a well-formed practical question comes up worth bringing to others (e.g. a training-instability issue specific to mixture density networks).

## Gaps

- No resource yet found specifically on MDN training pathologies (mixture collapse, exploding/vanishing sigma). Search for this if it shows up as a real problem in the user's training runs.
