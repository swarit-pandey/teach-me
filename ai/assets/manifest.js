// Single source of truth for course structure. index.html and lesson nav read this.
window.COURSE = {
  title: "AI from Scratch",
  parts: [
    { id: "P0", title: "Start here", lessons: [
      { n: "00", slug: "how-to-use", title: "How this course works", hours: 0.5 },
    ]},
    { id: "P1", title: "Linear algebra", blurb: "The language of data: vectors, transformations, and the geometry inside every model.", lessons: [
      { n: "01", slug: "vectors", title: "Vectors & the dot product", hours: 1.5 },
      { n: "02", slug: "matrices-as-transformations", title: "Matrices as transformations", hours: 1.5 },
      { n: "03", slug: "matmul-systems-inverse", title: "Matmul, systems, inverse & rank", hours: 2 },
      { n: "04", slug: "eigen-svd-pca", title: "Eigenvectors, SVD & PCA", hours: 2 },
    ]},
    { id: "P2", title: "Calculus", blurb: "How things change — and how to make a loss go down.", lessons: [
      { n: "05", slug: "derivatives-chain-rule", title: "Derivatives & the chain rule", hours: 1.5 },
      { n: "06", slug: "gradients-descent", title: "Partials, gradients & gradient descent", hours: 2 },
      { n: "07", slug: "jacobians-matrix-calculus", title: "Jacobians & matrix calculus", hours: 2 },
    ]},
    { id: "P3", title: "Probability & statistics", blurb: "Uncertainty, likelihood, and the information theory behind every loss function.", lessons: [
      { n: "08", slug: "probability-random-variables", title: "Probability & random variables", hours: 1.5 },
      { n: "09", slug: "expectation-gaussian-clt", title: "Expectation, variance, Gaussians & the CLT", hours: 1.5 },
      { n: "10", slug: "mle-bayes", title: "Maximum likelihood & Bayes", hours: 2 },
      { n: "11", slug: "entropy-cross-entropy-kl", title: "Entropy, cross-entropy & KL", hours: 1.5 },
    ]},
    { id: "P4", title: "Machine learning in code", blurb: "From math to working NumPy: your first models, trained from scratch.", lessons: [
      { n: "12", slug: "numpy-thinking", title: "NumPy thinking: vectorization & broadcasting", hours: 1.5 },
      { n: "13", slug: "linear-regression", title: "Linear regression from scratch", hours: 2 },
      { n: "14", slug: "logistic-softmax", title: "Logistic & softmax classification", hours: 2 },
      { n: "15", slug: "generalization", title: "Generalization, overfitting & regularization", hours: 1.5 },
    ]},
    { id: "P5", title: "Deep learning", blurb: "Stack the pieces: neural nets, backprop, and an autograd engine you build yourself.", lessons: [
      { n: "16", slug: "neural-networks", title: "Neural networks & MLPs", hours: 2 },
      { n: "17", slug: "backprop-autograd", title: "Backprop & building micro-autograd", hours: 2.5 },
      { n: "18", slug: "optimizers-init-norm", title: "Optimizers, initialization & normalization", hours: 2 },
      { n: "19", slug: "pytorch", title: "Moving to PyTorch", hours: 1.5 },
      { n: "20", slug: "embeddings-sequences", title: "Embeddings & sequence models", hours: 2 },
    ]},
    { id: "P6", title: "Transformers", blurb: "Attention, the transformer block, and a GPT you train on your laptop.", lessons: [
      { n: "21", slug: "tokenization", title: "Tokenization", hours: 1.5 },
      { n: "22", slug: "attention", title: "Attention", hours: 2.5 },
      { n: "23", slug: "transformer-block", title: "The transformer block", hours: 2 },
      { n: "24", slug: "tiny-gpt", title: "Train a tiny GPT", hours: 3 },
      { n: "25", slug: "sampling-scaling-finetuning", title: "Sampling, scaling & fine-tuning", hours: 2 },
    ]},
    { id: "P7", title: "RL on transformers", blurb: "Rewards, policy gradients, and how language models are aligned: RLHF, DPO, GRPO.", lessons: [
      { n: "26", slug: "rl-basics", title: "RL basics: MDPs, returns & bandits", hours: 2 },
      { n: "27", slug: "policy-gradients-ppo", title: "Policy gradients → PPO", hours: 2.5 },
      { n: "28", slug: "rlhf", title: "RLHF: reward models & PPO with KL", hours: 3 },
      { n: "29", slug: "dpo-grpo-rlvr", title: "DPO, GRPO & RL with verifiable rewards", hours: 2.5 },
    ]},
  ],
};
window.COURSE.all = window.COURSE.parts.flatMap(p => p.lessons.map(l => ({ ...l, part: p.id, partTitle: p.title })));
