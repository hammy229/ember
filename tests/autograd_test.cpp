// Checks every autograd operation's gradients against finite differences, in double precision, so a
// wrong backward rule shows up as a clear mismatch rather than as training that is merely worse.

#include "ember/autograd.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace {

namespace ag = ember::autograd;
using V = ag::Var<double>;

V Random(std::size_t rows, std::size_t cols, std::uint32_t seed, double center = 0) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(center, 1.0);
  std::vector<double> values(rows * cols);
  for (double& v : values) v = normal(rng);
  return ag::Leaf<double>(rows, cols, values);
}

// Reduces any output to a scalar with fixed random weights, so every output element matters.
V Reduce(const V& out, std::uint32_t seed = 99) {
  return ag::Sum(ag::Mul(out, Random(out->rows, out->cols, seed)));
}

// Compares Backward's gradient with a central difference for every element of every input.
void CheckGradients(const std::vector<V>& inputs, const std::function<V()>& f) {
  ag::ZeroGrad(inputs);
  ag::Backward(f());
  const double h = 1e-6;
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    for (std::size_t i = 0; i < inputs[k]->value.size(); ++i) {
      const double saved = inputs[k]->value[i];
      inputs[k]->value[i] = saved + h;
      const double up = f()->value[0];
      inputs[k]->value[i] = saved - h;
      const double down = f()->value[0];
      inputs[k]->value[i] = saved;
      const double numeric = (up - down) / (2 * h);
      const double analytic = inputs[k]->grad[i];
      EXPECT_NEAR(analytic, numeric, 1e-6 * std::max(1.0, std::abs(numeric)))
          << "input " << k << ", element " << i;
    }
  }
}

TEST(Autograd, MatMul) {
  const V a = Random(3, 4, 1), b = Random(4, 5, 2);
  CheckGradients({a, b}, [&] { return Reduce(ag::MatMul(a, b)); });
}

TEST(Autograd, Transpose) {
  const V a = Random(3, 5, 3);
  CheckGradients({a}, [&] { return Reduce(ag::Transpose(a)); });
}

TEST(Autograd, AddSameShapeAndBias) {
  const V a = Random(4, 3, 4), b = Random(4, 3, 5), bias = Random(1, 3, 6);
  CheckGradients({a, b}, [&] { return Reduce(ag::Add(a, b)); });
  CheckGradients({a, bias}, [&] { return Reduce(ag::Add(a, bias)); });
}

TEST(Autograd, ScaleMulAndSum) {
  const V a = Random(3, 3, 7), b = Random(3, 3, 8);
  CheckGradients({a}, [&] { return Reduce(ag::Scale(a, 0.37)); });
  CheckGradients({a, b}, [&] { return Reduce(ag::Mul(a, b)); });
  CheckGradients({a}, [&] { return ag::Sum(a); });
}

TEST(Autograd, ReluAwayFromTheKink) {
  V a = Random(4, 4, 9);
  for (double& v : a->value) v += v > 0 ? 0.1 : -0.1;  // finite differences cannot cross 0
  CheckGradients({a}, [&] { return Reduce(ag::Relu(a)); });
}

TEST(Autograd, CausalSoftmax) {
  const V a = Random(5, 5, 10);
  CheckGradients({a}, [&] { return Reduce(ag::CausalSoftmax(a)); });
  // Each row sums to 1 over its visible columns, and nothing leaks from the future.
  const V y = ag::CausalSoftmax(a);
  for (std::size_t i = 0; i < 5; ++i) {
    double total = 0;
    for (std::size_t j = 0; j < 5; ++j) {
      if (j > i) EXPECT_EQ(y->at(i, j), 0.0);
      total += y->at(i, j);
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
  }
}

TEST(Autograd, RmsNorm) {
  const V a = Random(3, 6, 11), w = Random(1, 6, 12, 1.0);
  CheckGradients({a, w}, [&] { return Reduce(ag::RmsNorm(a, w)); });
}

TEST(Autograd, EmbeddingWithRepeatedIds) {
  const V table = Random(6, 4, 13);
  const std::vector<std::int32_t> ids{2, 5, 2, 0};
  CheckGradients({table}, [&] { return Reduce(ag::Embedding(table, std::span(ids))); });
}

TEST(Autograd, RowsColumnsAndConcat) {
  const V a = Random(6, 8, 14);
  CheckGradients({a}, [&] { return Reduce(ag::TopRows(a, 4)); });
  CheckGradients({a}, [&] { return Reduce(ag::Columns(a, 2, 3)); });
  CheckGradients({a}, [&] {
    return Reduce(ag::ConcatColumns<double>({ag::Columns(a, 4, 4), ag::Columns(a, 0, 4)}));
  });
}

TEST(Autograd, CrossEntropy) {
  const V logits = Random(4, 7, 15);
  const std::vector<std::int32_t> targets{3, 0, 6, 3};
  CheckGradients({logits}, [&] { return ag::CrossEntropy(logits, std::span(targets)); });
  // Uniform logits cost log(vocab).
  const V flat = ag::Leaf<double>(2, 7);
  const std::vector<std::int32_t> two{1, 4};
  EXPECT_NEAR(ag::CrossEntropy(flat, std::span(two))->value[0], std::log(7.0), 1e-12);
}

// One attention head end to end, the way the model uses these pieces.
TEST(Autograd, AWholeAttentionHead) {
  const V x = Random(5, 8, 16), wq = Random(8, 4, 17), wk = Random(8, 4, 18), wv = Random(8, 4, 19);
  CheckGradients({x, wq, wk, wv}, [&] {
    const V q = ag::MatMul(x, wq), k = ag::MatMul(x, wk), v = ag::MatMul(x, wv);
    const V att = ag::CausalSoftmax(ag::Scale(ag::MatMul(q, ag::Transpose(k)), 0.5));
    return Reduce(ag::MatMul(att, v));
  });
}

// A value used twice gets both gradients added, not one overwriting the other.
TEST(Autograd, SharedSubexpressionsAddTheirGradients) {
  const V a = Random(2, 2, 20);
  CheckGradients({a}, [&] {
    const V b = ag::Scale(a, 3.0);
    return ag::Sum(ag::Add(ag::Mul(b, b), b));
  });
}

TEST(Autograd, AdamWFindsTheMinimumOfAQuadratic) {
  const V w = Random(1, 5, 21);
  const V target = Random(1, 5, 22);
  ag::AdamW<double> optimizer({w}, 0.05, 0.0);
  double loss = 0;
  for (int step = 0; step < 2000; ++step) {
    ag::ZeroGrad<double>({w});
    const V diff = ag::Add(w, ag::Scale(target, -1.0));
    const V l = ag::Sum(ag::Mul(diff, diff));
    loss = l->value[0];
    ag::Backward(l);
    optimizer.Step();
  }
  EXPECT_LT(loss, 1e-6);
}

}  // namespace
