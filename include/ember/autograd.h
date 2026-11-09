#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

namespace ember::autograd {

// Reverse-mode automatic differentiation over small dense matrices, enough to train a tiny
// transformer. Every operation records how to send gradients back to its inputs; Backward walks
// that record in reverse.
//
// It is a template on the number type: training uses float, and tests/autograd_test.cpp checks
// every operation's gradients against finite differences in double, where the check can be tight.
//
// Matrices are row-major [rows, cols]. A scalar is [1, 1].
template <typename T>
struct Node {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<T> value;
  std::vector<T> grad;
  std::vector<std::shared_ptr<Node>> parents;
  // Adds this node's gradient into its parents' gradients.
  std::function<void(Node&)> backward;

  T& at(std::size_t r, std::size_t c) { return value[r * cols + c]; }
  T at(std::size_t r, std::size_t c) const { return value[r * cols + c]; }
};

template <typename T>
using Var = std::shared_ptr<Node<T>>;

// A matrix with no history: a parameter, or an input.
template <typename T>
Var<T> Leaf(std::size_t rows, std::size_t cols, std::vector<T> value = {}) {
  auto node = std::make_shared<Node<T>>();
  node->rows = rows;
  node->cols = cols;
  node->value = value.empty() ? std::vector<T>(rows * cols, T(0)) : std::move(value);
  node->grad.assign(rows * cols, T(0));
  return node;
}

namespace detail {

template <typename T>
Var<T> Result(std::size_t rows, std::size_t cols, std::vector<Var<T>> parents) {
  auto node = Leaf<T>(rows, cols);
  node->parents = std::move(parents);
  return node;
}

}  // namespace detail

// a [n, k] times b [k, m].
template <typename T>
Var<T> MatMul(const Var<T>& a, const Var<T>& b) {
  const std::size_t n = a->rows, k = a->cols, m = b->cols;
  auto out = detail::Result<T>(n, m, {a, b});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t p = 0; p < k; ++p) {
      const T av = a->value[i * k + p];
      const T* brow = &b->value[p * m];
      T* orow = &out->value[i * m];
      for (std::size_t j = 0; j < m; ++j) orow[j] += av * brow[j];
    }
  }
  out->backward = [n, k, m](Node<T>& o) {
    Node<T>& a = *o.parents[0];
    Node<T>& b = *o.parents[1];
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t p = 0; p < k; ++p) {
        const T* g = &o.grad[i * m];
        const T* brow = &b.value[p * m];
        T* bgrad = &b.grad[p * m];
        const T av = a.value[i * k + p];
        T sum = 0;
        for (std::size_t j = 0; j < m; ++j) {
          sum += g[j] * brow[j];
          bgrad[j] += av * g[j];
        }
        a.grad[i * k + p] += sum;
      }
    }
  };
  return out;
}

template <typename T>
Var<T> Transpose(const Var<T>& a) {
  auto out = detail::Result<T>(a->cols, a->rows, {a});
  for (std::size_t r = 0; r < a->rows; ++r) {
    for (std::size_t c = 0; c < a->cols; ++c) out->at(c, r) = a->at(r, c);
  }
  out->backward = [](Node<T>& o) {
    Node<T>& a = *o.parents[0];
    for (std::size_t r = 0; r < a.rows; ++r) {
      for (std::size_t c = 0; c < a.cols; ++c) a.grad[r * a.cols + c] += o.grad[c * o.cols + r];
    }
  };
  return out;
}

// Same shapes, or b a single row added to every row of a (a bias).
template <typename T>
Var<T> Add(const Var<T>& a, const Var<T>& b) {
  const bool row = b->rows == 1 && a->rows != 1;
  auto out = detail::Result<T>(a->rows, a->cols, {a, b});
  for (std::size_t i = 0; i < out->value.size(); ++i) {
    out->value[i] = a->value[i] + b->value[row ? i % a->cols : i];
  }
  out->backward = [row](Node<T>& o) {
    Node<T>& a = *o.parents[0];
    Node<T>& b = *o.parents[1];
    for (std::size_t i = 0; i < o.grad.size(); ++i) {
      a.grad[i] += o.grad[i];
      b.grad[row ? i % o.cols : i] += o.grad[i];
    }
  };
  return out;
}

template <typename T>
Var<T> Scale(const Var<T>& a, T s) {
  auto out = detail::Result<T>(a->rows, a->cols, {a});
  for (std::size_t i = 0; i < out->value.size(); ++i) out->value[i] = a->value[i] * s;
  out->backward = [s](Node<T>& o) {
    for (std::size_t i = 0; i < o.grad.size(); ++i) o.parents[0]->grad[i] += o.grad[i] * s;
  };
  return out;
}

// Element by element, same shapes.
template <typename T>
Var<T> Mul(const Var<T>& a, const Var<T>& b) {
  auto out = detail::Result<T>(a->rows, a->cols, {a, b});
  for (std::size_t i = 0; i < out->value.size(); ++i) out->value[i] = a->value[i] * b->value[i];
  out->backward = [](Node<T>& o) {
    Node<T>& a = *o.parents[0];
    Node<T>& b = *o.parents[1];
    for (std::size_t i = 0; i < o.grad.size(); ++i) {
      a.grad[i] += o.grad[i] * b.value[i];
      b.grad[i] += o.grad[i] * a.value[i];
    }
  };
  return out;
}

// Every element added up, as a scalar.
template <typename T>
Var<T> Sum(const Var<T>& a) {
  auto out = detail::Result<T>(1, 1, {a});
  for (const T v : a->value) out->value[0] += v;
  out->backward = [](Node<T>& o) {
    for (T& g : o.parents[0]->grad) g += o.grad[0];
  };
  return out;
}

template <typename T>
Var<T> Relu(const Var<T>& a) {
  auto out = detail::Result<T>(a->rows, a->cols, {a});
  for (std::size_t i = 0; i < out->value.size(); ++i) {
    out->value[i] = a->value[i] > T(0) ? a->value[i] : T(0);
  }
  out->backward = [](Node<T>& o) {
    Node<T>& a = *o.parents[0];
    for (std::size_t i = 0; i < o.grad.size(); ++i) {
      if (a.value[i] > T(0)) a.grad[i] += o.grad[i];
    }
  };
  return out;
}

// Softmax over each row with the future hidden: row i only spreads over columns 0 to i. This is
// attention's mask and softmax in one step.
template <typename T>
Var<T> CausalSoftmax(const Var<T>& a) {
  const std::size_t n = a->rows, m = a->cols;
  auto out = detail::Result<T>(n, m, {a});
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t last = std::min(i, m - 1);
    T largest = a->at(i, 0);
    for (std::size_t j = 1; j <= last; ++j) largest = std::max(largest, a->at(i, j));
    T sum = 0;
    for (std::size_t j = 0; j <= last; ++j)
      sum += (out->at(i, j) = std::exp(a->at(i, j) - largest));
    for (std::size_t j = 0; j <= last; ++j) out->at(i, j) /= sum;
  }
  out->backward = [n, m](Node<T>& o) {
    Node<T>& a = *o.parents[0];
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t last = std::min(i, m - 1);
      T dot = 0;
      for (std::size_t j = 0; j <= last; ++j) dot += o.grad[i * m + j] * o.value[i * m + j];
      for (std::size_t j = 0; j <= last; ++j) {
        a.grad[i * m + j] += o.value[i * m + j] * (o.grad[i * m + j] - dot);
      }
    }
  };
  return out;
}

// RMSNorm over each row, times weight [1, cols].
template <typename T>
Var<T> RmsNorm(const Var<T>& a, const Var<T>& weight, T eps = T(1e-5)) {
  const std::size_t n = a->rows, m = a->cols;
  auto out = detail::Result<T>(n, m, {a, weight});
  auto inverse = std::make_shared<std::vector<T>>(n);
  for (std::size_t i = 0; i < n; ++i) {
    T squares = 0;
    for (std::size_t j = 0; j < m; ++j) squares += a->at(i, j) * a->at(i, j);
    (*inverse)[i] = T(1) / std::sqrt(squares / T(m) + eps);
    for (std::size_t j = 0; j < m; ++j) {
      out->at(i, j) = a->at(i, j) * (*inverse)[i] * weight->value[j];
    }
  }
  out->backward = [n, m, inverse](Node<T>& o) {
    Node<T>& a = *o.parents[0];
    Node<T>& w = *o.parents[1];
    for (std::size_t i = 0; i < n; ++i) {
      const T r = (*inverse)[i];
      // y_j = x_j r w_j, and r = (mean(x^2) + eps)^(-1/2), so dr/dx_k = -r^3 x_k / m.
      T dot = 0;
      for (std::size_t j = 0; j < m; ++j) {
        const T g = o.grad[i * m + j];
        w.grad[j] += g * a.value[i * m + j] * r;
        dot += g * w.value[j] * a.value[i * m + j];
      }
      for (std::size_t k = 0; k < m; ++k) {
        a.grad[i * m + k] +=
            o.grad[i * m + k] * w.value[k] * r - dot * r * r * r * a.value[i * m + k] / T(m);
      }
    }
  };
  return out;
}

// Rows of table [vocab, cols] picked by ids, giving [ids.size(), cols].
template <typename T>
Var<T> Embedding(const Var<T>& table, std::span<const std::int32_t> ids) {
  const std::size_t m = table->cols;
  auto out = detail::Result<T>(ids.size(), m, {table});
  auto rows = std::make_shared<std::vector<std::int32_t>>(ids.begin(), ids.end());
  for (std::size_t i = 0; i < ids.size(); ++i) {
    std::copy_n(&table->value[static_cast<std::size_t>(ids[i]) * m], m, &out->value[i * m]);
  }
  out->backward = [m, rows](Node<T>& o) {
    Node<T>& t = *o.parents[0];
    for (std::size_t i = 0; i < rows->size(); ++i) {
      T* g = &t.grad[static_cast<std::size_t>((*rows)[i]) * m];
      for (std::size_t j = 0; j < m; ++j) g[j] += o.grad[i * m + j];
    }
  };
  return out;
}

// The first `count` rows of a: positions 0 to count - 1 of a position table.
template <typename T>
Var<T> TopRows(const Var<T>& a, std::size_t count) {
  auto out = detail::Result<T>(count, a->cols, {a});
  std::copy_n(a->value.begin(), count * a->cols, out->value.begin());
  out->backward = [](Node<T>& o) {
    for (std::size_t i = 0; i < o.grad.size(); ++i) o.parents[0]->grad[i] += o.grad[i];
  };
  return out;
}

// Columns [start, start + count) of a: one attention head's slice.
template <typename T>
Var<T> Columns(const Var<T>& a, std::size_t start, std::size_t count) {
  auto out = detail::Result<T>(a->rows, count, {a});
  for (std::size_t r = 0; r < a->rows; ++r) {
    std::copy_n(&a->value[r * a->cols + start], count, &out->value[r * count]);
  }
  out->backward = [start, count](Node<T>& o) {
    Node<T>& a = *o.parents[0];
    for (std::size_t r = 0; r < a.rows; ++r) {
      for (std::size_t c = 0; c < count; ++c)
        a.grad[r * a.cols + start + c] += o.grad[r * count + c];
    }
  };
  return out;
}

// Side by side: heads back into one matrix.
template <typename T>
Var<T> ConcatColumns(const std::vector<Var<T>>& parts) {
  std::size_t cols = 0;
  for (const auto& p : parts) cols += p->cols;
  const std::size_t rows = parts.front()->rows;
  auto out = detail::Result<T>(rows, cols, parts);
  std::size_t offset = 0;
  for (const auto& p : parts) {
    for (std::size_t r = 0; r < rows; ++r) {
      std::copy_n(&p->value[r * p->cols], p->cols, &out->value[r * cols + offset]);
    }
    offset += p->cols;
  }
  out->backward = [rows, cols](Node<T>& o) {
    std::size_t offset = 0;
    for (const auto& p : o.parents) {
      for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < p->cols; ++c)
          p->grad[r * p->cols + c] += o.grad[r * cols + offset + c];
      }
      offset += p->cols;
    }
  };
  return out;
}

// Mean cross-entropy of logits [n, vocab] against the right next token for each row, as a scalar.
template <typename T>
Var<T> CrossEntropy(const Var<T>& logits, std::span<const std::int32_t> targets) {
  const std::size_t n = logits->rows, v = logits->cols;
  auto out = detail::Result<T>(1, 1, {logits});
  auto probs = std::make_shared<std::vector<T>>(n * v);
  auto right = std::make_shared<std::vector<std::int32_t>>(targets.begin(), targets.end());
  T loss = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const T* row = &logits->value[i * v];
    const T largest = *std::max_element(row, row + v);
    T sum = 0;
    for (std::size_t j = 0; j < v; ++j) sum += ((*probs)[i * v + j] = std::exp(row[j] - largest));
    for (std::size_t j = 0; j < v; ++j) (*probs)[i * v + j] /= sum;
    loss -= std::log((*probs)[i * v + static_cast<std::size_t>(targets[i])]);
  }
  out->value[0] = loss / T(n);
  out->backward = [n, v, probs, right](Node<T>& o) {
    Node<T>& l = *o.parents[0];
    const T scale = o.grad[0] / T(n);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < v; ++j) {
        T g = (*probs)[i * v + j];
        if (j == static_cast<std::size_t>((*right)[i])) g -= T(1);
        l.grad[i * v + j] += scale * g;
      }
    }
  };
  return out;
}

// Sends gradients from a scalar back through everything it was computed from. Gradients add up,
// so call ZeroGrad on the parameters before each step.
template <typename T>
void Backward(const Var<T>& root) {
  std::vector<Node<T>*> order;
  std::unordered_set<Node<T>*> seen;
  // Depth-first, children before parents, without recursion so long graphs cannot overflow the
  // stack.
  std::vector<std::pair<Node<T>*, std::size_t>> stack{{root.get(), 0}};
  seen.insert(root.get());
  while (!stack.empty()) {
    auto& [node, next] = stack.back();
    if (next < node->parents.size()) {
      Node<T>* parent = node->parents[next++].get();
      if (seen.insert(parent).second) stack.push_back({parent, 0});
    } else {
      order.push_back(node);
      stack.pop_back();
    }
  }
  root->grad.assign(root->grad.size(), T(1));
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    if ((*it)->backward) (*it)->backward(**it);
  }
}

template <typename T>
void ZeroGrad(const std::vector<Var<T>>& parameters) {
  for (const auto& p : parameters) std::fill(p->grad.begin(), p->grad.end(), T(0));
}

// AdamW: Adam with weight decay applied to the weights directly, as in Loshchilov and Hutter.
template <typename T>
class AdamW {
 public:
  explicit AdamW(std::vector<Var<T>> parameters, T rate = T(1e-3), T decay = T(0.01),
                 T beta1 = T(0.9), T beta2 = T(0.999), T eps = T(1e-8))
      : parameters_(std::move(parameters)),
        rate_(rate),
        decay_(decay),
        beta1_(beta1),
        beta2_(beta2),
        eps_(eps) {
    for (const auto& p : parameters_) {
      m_.emplace_back(p->value.size(), T(0));
      v_.emplace_back(p->value.size(), T(0));
    }
  }

  void Step() {
    ++step_;
    const T c1 = T(1) - std::pow(beta1_, T(step_));
    const T c2 = T(1) - std::pow(beta2_, T(step_));
    for (std::size_t k = 0; k < parameters_.size(); ++k) {
      Node<T>& p = *parameters_[k];
      for (std::size_t i = 0; i < p.value.size(); ++i) {
        const T g = p.grad[i];
        m_[k][i] = beta1_ * m_[k][i] + (T(1) - beta1_) * g;
        v_[k][i] = beta2_ * v_[k][i] + (T(1) - beta2_) * g * g;
        p.value[i] -=
            rate_ * ((m_[k][i] / c1) / (std::sqrt(v_[k][i] / c2) + eps_) + decay_ * p.value[i]);
      }
    }
  }

  void set_rate(T rate) { rate_ = rate; }

 private:
  std::vector<Var<T>> parameters_;
  std::vector<std::vector<T>> m_;
  std::vector<std::vector<T>> v_;
  T rate_, decay_, beta1_, beta2_, eps_;
  std::size_t step_ = 0;
};

}  // namespace ember::autograd
